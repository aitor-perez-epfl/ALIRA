import logging
import time

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans

from alira.classifiers import LogisticRegressionClassifier
from alira.llms import generate_documents, evaluate_documents, send_embedding_request

logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

_N_FORMAT_EXAMPLES = 5


def select_stratified_diverse(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Stratified by confidence + diverse within each stratum."""
    if len(df) == 0 or n_samples <= 0:
        return df.head(0)

    df = df.copy()

    zones = [
        (df[df["prediction"] > 0.7], 0.4),  # 40% high confidence positive
        (df[df["prediction"].between(0.3, 0.7)], 0.4),  # 40% uncertain
        (df[df["prediction"] < 0.3], 0.2),  # 20% likely negative
    ]

    selected = []
    for zone_df, fraction in zones:
        if len(zone_df) == 0:
            continue

        n_zone = max(1, int(n_samples * fraction))

        # Use clustering for diversity within zone
        n_clusters = min(n_zone, len(zone_df))
        if n_clusters > 1:
            embeddings = np.vstack(zone_df["embedding"].values)
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
            zone_df = zone_df.copy()
            zone_df["cluster"] = kmeans.fit_predict(embeddings)

            # Random sample from each cluster
            for c in range(n_clusters):
                cluster = zone_df[zone_df["cluster"] == c]
                if len(cluster) > 0:
                    selected.append(cluster.sample(1).index[0])
        else:
            selected.extend(zone_df.sample(min(n_zone, len(zone_df))).index)

    return df.loc[selected[:n_samples]]  # Trim to exact budget


def select_candidates_to_evaluate(
    df: pd.DataFrame, n_samples: int, cluster: bool = False
) -> pd.DataFrame:
    """Select candidates for evaluation using a confidence-based stratified sampling strategy.

    This function divides the input data into three confidence zones based on the
    'prediction' column and samples within each zone to ensure a balanced representation
    of high-confidence positives, uncertain items, and likely negatives.

    The sampling budget is allocated as follows:
        - High confidence positive (prediction > 0.7): 30%
        - Uncertain (prediction between 0.3 and 0.7): 40%
        - Likely negative (prediction < 0.3): 30%

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'prediction' column.
            The index values are used to select the returned rows.
        n_samples (int): Total number of samples to select.
        cluster (bool): When False, samples randomly within each stratum. When True,
            uses MiniBatchKMeans clustering on the 'embedding' column to pick one
            item per cluster for diversity.
            Default is False.

    Returns:
        pd.DataFrame: A DataFrame containing the selected candidate rows. If the
            input is empty or n_samples is <= 0, returns an empty DataFrame.
    """
    if len(df) == 0 or n_samples <= 0:
        return df.head(0)

    zones = [
        (df[df["prediction"] > 0.7], 0.3),       # 30% high confidence positive
        (df[df["prediction"].between(0.3, 0.7)], 0.4),  # 40% uncertain
        (df[df["prediction"] < 0.3], 0.3),       # 30% likely negative
    ]

    selected = []
    for zone_df, fraction in zones:
        if len(zone_df) == 0:
            continue

        n_zone = max(1, int(n_samples * fraction))

        if cluster:
            n_clusters = min(n_zone, len(zone_df))
            if n_clusters > 1:
                embeddings = np.vstack(zone_df["embedding"].values)
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
                zone_df = zone_df.copy()
                zone_df["cluster"] = kmeans.fit_predict(embeddings)

                for c in range(n_clusters):
                    cluster_items = zone_df[zone_df["cluster"] == c]
                    if len(cluster_items) > 0:
                        selected.append(cluster_items.sample(1).index[0])
            else:
                selected.extend(zone_df.sample(min(n_zone, len(zone_df))).index)
        else:
            selected.extend(zone_df.sample(min(n_zone, len(zone_df))).index)

    return df.loc[selected[:n_samples]]  # Trim to exact budget


class ActiveLearner:
    """Active learning classifier for document filtering."""

    def __init__(
        self,
        n_synthetic_documents: int = 10,
        min_iterations: int = 3,
        max_iterations: int = 20,
        n_eval_per_iteration: int = 30,
        c_value: float = 1.0,
        early_stop_threshold: float = 0.02,
        uncertain_rmse_early_stop_threshold: float = 0.01,
        pos_rmse_early_stop_threshold: float = 0.01,
        generation_prompt: str | None = None,
        evaluation_prompt: str | None = None,
    ):
        """
        Args:
            n_synthetic_documents: Number of synthetic documents to generate for bootstrapping
            min_iterations: Minimum iterations before early stopping is evaluated
            max_iterations: Maximum active learning iterations
            n_eval_per_iteration: Documents evaluated per iteration
            c_value: C parameter for LogisticRegression
            early_stop_threshold: Max flip rate to consider predictions stable
            uncertain_rmse_early_stop_threshold: Max RMSE in the uncertain zone (0.3–0.7) to consider stable
            pos_rmse_early_stop_threshold: Max RMSE in the positive zone (>=0.5) to consider stable
            generation_prompt: Replaces the default synthetic document generation prompt
            evaluation_prompt: Replaces the default document evaluation prompt
        """
        self.n_synthetic_documents = n_synthetic_documents
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.n_eval_per_iteration = n_eval_per_iteration
        self.c_value = c_value
        self.early_stop_threshold = early_stop_threshold
        self.uncertain_rmse_early_stop_threshold = uncertain_rmse_early_stop_threshold
        self.pos_rmse_early_stop_threshold = pos_rmse_early_stop_threshold
        self.generation_prompt = generation_prompt
        self.evaluation_prompt = evaluation_prompt

    def fit(self, df: pd.DataFrame, query: str):
        """
        Run active learning classification.

        Args:
            df: Corpus with a required `text` column and an optional `embedding` column.
                If embeddings are absent they are generated at the start of fit.
            query: Search query/topic

        Returns:
            results_df (positive items with scores), params_dict
        """
        start_time = time.time()
        query = query.strip()[:15000]

        logger.info("Starting classification for query: %s", query)

        df = df.copy()
        if "embedding" not in df.columns:
            logger.info("Generating embeddings for %s documents...", len(df))
            df["embedding"] = send_embedding_request(df["text"].tolist())
            logger.info("Embeddings generated.")

        non_empty = df[df["text"].str.strip() != ""]["text"]
        format_examples = non_empty.sample(min(_N_FORMAT_EXAMPLES, len(non_empty))).tolist()

        logger.info("Generating %s synthetic documents...", self.n_synthetic_documents)
        synthetic_documents = generate_documents(
            query, self.n_synthetic_documents, format_examples, self.generation_prompt
        )
        logger.info("Generated %s synthetic documents", len(synthetic_documents))
        logger.info(synthetic_documents)

        logger.info("Embedding synthetic documents...")
        synthetic_embeddings = send_embedding_request(synthetic_documents)
        synthetic_embeddings = np.array(synthetic_embeddings)
        logger.info("Embedded %s synthetic documents", len(synthetic_embeddings))

        n_documents = len(df)
        original_columns = list(df.columns)

        documents_df = df
        documents_df["gt"] = pd.NA
        documents_df["is_synthetic"] = False

        synthetic_df = pd.DataFrame({
            "text": synthetic_documents,
            "embedding": [synthetic_embeddings[i] for i in range(len(synthetic_documents))],
            "is_synthetic": True,
            "gt": True
        })
        synthetic_df.index = range(-1, -1 - len(synthetic_df), -1)

        # Combine both dataframes
        df = pd.concat([synthetic_df, documents_df])

        # Compute distance to synthetic centroid
        all_embeddings = np.vstack(df["embedding"].values)
        synthetic_centroid = np.mean(synthetic_embeddings, axis=0).reshape(1, -1)
        synthetic_centroid = synthetic_centroid / np.linalg.norm(synthetic_centroid)
        df["cosine_similarity"] = all_embeddings @ synthetic_centroid.T

        # A priori probabilities are linear and evenly distributed on [0, 1] based on the cosine_similarity order
        df["prediction"] = df['cosine_similarity'].rank() / len(df)
        df["prediction_binary"] = df["prediction"] > 0.5
        df["confidence"] = (df["prediction"] - 0.5).abs()

        not_is_synthetic = ~df["is_synthetic"]

        # print(f"Selecting the closest {self.n_nearest_start} documents to the synthetic centroid as candidates to evaluate...")
        # candidates = df.loc[not_is_synthetic].nsmallest(self.n_nearest_start, "distance_to_centroid")
        candidates = select_candidates_to_evaluate(df.loc[not_is_synthetic], self.n_eval_per_iteration)

        # Active Learning loop
        logger.info("Starting active learning loop...")
        classifier = None
        prev_positives = None
        prev_predictions = None
        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            # Evaluate candidates with LLM
            logger.info("Iteration %s: Evaluating %s documents...", iteration, len(candidates))
            evaluations = evaluate_documents(query=query, texts=candidates["text"].tolist(), prompt=self.evaluation_prompt)
            df.loc[candidates.index, "gt"] = pd.array(evaluations, dtype="boolean")

            yes_count = df.loc[candidates.index].query("gt == True").shape[0]
            no_count = df.loc[candidates.index].query("gt == False").shape[0]
            logger.info("Iteration %s: Evaluated %s documents. Yes: %s, No: %s", iteration, len(candidates), yes_count, no_count)

            # Train on labeled data
            training_df = df.dropna(subset=["gt"]).copy()
            training_df["gt"] = training_df["gt"].astype(bool)

            X_train = np.vstack(training_df["embedding"].values)
            y_train = training_df["gt"].values

            # Check we have both classes
            if len(np.unique(y_train)) < 2:
                # Add farthest unlabeled as negatives
                unlabeled = df[not_is_synthetic & df["gt"].isna()]
                if len(unlabeled) > 0:
                    n_add = max(1, int(y_train.sum()))
                    farthest = unlabeled.nsmallest(n_add, "cosine_similarity")
                    df.loc[farthest.index, "gt"] = False
                    training_df = df.dropna(subset=["gt"]).copy()
                    training_df["gt"] = training_df["gt"].astype(bool)
                    X_train = np.vstack(training_df["embedding"].values)
                    y_train = training_df["gt"].values
                    logger.info("Added %s distant docs as negatives", len(farthest))

            if len(np.unique(y_train)) < 2:
                logger.info("Iteration %s: Skipping - need both classes", iteration)
                continue

            # Train classifier
            dist_dict = training_df["gt"].value_counts().to_dict()
            logger.info("Iteration %s: Training classifier on %s documents (%s) labeled documents...", iteration, len(y_train), dist_dict)
            classifier = LogisticRegressionClassifier(c=self.c_value)
            classifier.fit(X_train, y_train)
            logger.info("Iteration %s: Trained classifier on %s documents (%s) labeled documents.", iteration, len(y_train), dist_dict)

            # Predict
            logger.info("Iteration %s: Predicting all documents with freshly trained classifier...", iteration)
            df["prediction"] = classifier.predict_proba(all_embeddings)[:, -1]
            df["prediction_binary"] = df["prediction"] > 0.5
            df["confidence"] = (df["prediction"] - 0.5).abs()
            pred_dict = df.loc[not_is_synthetic, "prediction_binary"].value_counts().to_dict()
            logger.info("Iteration %s: Predicted all documents: %s", iteration, pred_dict)

            # Early stopping
            positives = set(df.index[not_is_synthetic & df["prediction_binary"]])
            predictions = df.loc[not_is_synthetic, 'prediction'].values
            if prev_positives is not None and iteration > self.min_iterations:
                flipped = len(positives ^ prev_positives)
                total = len(positives | prev_positives)
                flip_rate = flipped / total if total > 0 else 0
                logger.info("Flip rate: %.2f%%", flip_rate * 100)

                uncertain = (
                    ((predictions >= 0.3) & (predictions <= 0.7)) |
                    ((prev_predictions >= 0.3) & (prev_predictions <= 0.7))
                )
                pos = (
                    (predictions >= 0.5) | (prev_predictions >= 0.5)
                )
                uncertain_rmse = np.sqrt(np.mean((predictions[uncertain] - prev_predictions[uncertain]) ** 2)) if uncertain.sum() > 0 else 0.0
                pos_rmse = np.sqrt(np.mean((predictions[pos] - prev_predictions[pos]) ** 2)) if pos.sum() > 0 else 0.0
                logger.info("Uncertain zone RMSE: %.4f", uncertain_rmse)
                logger.info("Positive zone RMSE: %.4f", pos_rmse)

                if flip_rate < self.early_stop_threshold or uncertain_rmse < self.uncertain_rmse_early_stop_threshold or pos_rmse < self.pos_rmse_early_stop_threshold:
                    logger.info("Early stop (flip-rate: %.2f%%, uncertain RMSE: %.4f, pos RMSE: %.4f)", flip_rate * 100, uncertain_rmse, pos_rmse)
                    break
            prev_positives = positives.copy()
            prev_predictions = predictions.copy()

            # Select next candidates
            if iteration < self.max_iterations:
                logger.info("Iteration %s: Selecting candidates for next iteration...", iteration)
                unlabeled = df[not_is_synthetic & df["gt"].isna()]
                if len(unlabeled) == 0:
                    logger.info("All documents labeled, stopping.")
                    break

                candidates = select_candidates_to_evaluate(unlabeled, self.n_eval_per_iteration)
                if len(candidates) == 0:
                    logger.info("No candidates found, stopping.")
                    break
                logger.info("Iteration %s: Selected %s candidates to evaluate...", iteration, len(candidates))

        # Keep only real documents with a positive prediction
        positives = df.loc[not_is_synthetic & df['prediction_binary']]

        # Add score, prediction_binary and confidence columns from the classifier
        results_df = positives[original_columns + ["prediction", "prediction_binary", "confidence"]].copy()
        results_df = results_df.rename(columns={"prediction": "score"})

        # Sort by score
        results_df = results_df.sort_values("score", ascending=False)

        # Save parameters
        elapsed = time.time() - start_time
        params = {
            "query": query,
            "n_synthetic_documents": self.n_synthetic_documents,
            "min_iterations": self.min_iterations,
            "max_iterations": self.max_iterations,
            "n_eval_per_iteration": self.n_eval_per_iteration,
            "execution_times": {
                "total_seconds": elapsed
            },
            "statistics": {
                "total_items": n_documents,
                "positive_items": len(results_df),
                "negative_items": n_documents - len(results_df),
                "iterations_completed": iteration
            },
            "model_info": {
                "type": "LogisticRegression",
                "c_value": self.c_value
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        logger.info("Done! Time: %.2fs", elapsed)

        return results_df, params
