import logging
import time

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans

from alira.classifiers import LogisticRegressionClassifier
from alira.synthetic import generate_synthetic_texts
from alira.evaluation import evaluate
from alira.llms import send_embedding_request

logger = logging.getLogger(__name__)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

_N_FORMAT_EXAMPLES = 5


class ActiveLearner:
    """Active-learning binary classifier that bootstraps with LLM-generated synthetic texts.

    Iteratively evaluates corpus items via an LLM, trains a Logistic Regression model on the
    labels, and stops early when the positive-zone prediction drift (RMSE) falls below a
    threshold. Returns items classified as positive, ranked by predicted score.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_synthetic: int = 10,
        min_iterations: int = 3,
        max_iterations: int = 20,
        n_eval_per_iteration: int = 30,
        c_value: float = 1.0,
        positive_zone_rmse_threshold: float = 0.01,
        cluster_candidates: bool = False,
        generation_prompt: str | None = None,
        evaluation_prompt: str | None = None,
    ):
        """
        Args:
            df: Corpus with a required ``text`` column and optional ``embedding`` column
            n_synthetic: Number of synthetic texts to generate for bootstrapping
            min_iterations: Minimum iterations before early stopping is evaluated
            max_iterations: Maximum active learning iterations
            n_eval_per_iteration: Number of texts evaluated per iteration
            c_value: C parameter for LogisticRegression
            positive_zone_rmse_threshold: Max RMSE in the positive zone (>=0.5) to consider stable
            cluster_candidates: If True, cluster candidates within each stratum for diversity
            generation_prompt: Replaces the default synthetic text generation prompt
            evaluation_prompt: Replaces the default text evaluation prompt
        """
        df = df.copy()
        # Extract embeddings if present so user df stays clean; store as numpy array
        self.embeddings_ = np.vstack(df.pop("embedding").values) if "embedding" in df.columns else None
        self.corpus_ = df
        self.n_corpus_ = len(self.corpus_)
        self.n_synthetic = n_synthetic
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.n_eval_per_iteration = n_eval_per_iteration
        self.c_value = c_value
        self.positive_zone_rmse_threshold = positive_zone_rmse_threshold
        self.cluster_candidates = cluster_candidates
        self.generation_prompt = generation_prompt
        self.evaluation_prompt = evaluation_prompt

        # Fit-time attributes (sklearn convention: trailing underscore)
        self.classifier_ = None
        self.query_ = None
        self.iterations_ = None
        self.execution_time_ = None

    @property
    def embeddings(self) -> np.ndarray:
        """Return cached embeddings or compute and cache them."""
        if self.embeddings_ is None:
            logger.info("Generating embeddings for %s texts...", self.n_corpus_)
            self.embeddings_ = np.array(send_embedding_request(self.corpus_["text"].tolist()))
            logger.info("Embeddings generated.")
        return self.embeddings_

    def _select_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select candidates from a real (non-synthetic) subset using stratified sampling."""
        n_samples = self.n_eval_per_iteration

        if len(df) == 0 or n_samples <= 0:
            return df.head(0)

        zones = [
            (df[df["prediction"] > 0.7], 0.30),
            (df[df["prediction"].between(0.3, 0.7)], 0.40),
            (df[df["prediction"] < 0.3], 0.30),
        ]

        selected = []
        for zone_df, fraction in zones:
            if len(zone_df) == 0:
                continue

            n_zone = max(1, int(n_samples * fraction))

            if self.cluster_candidates:
                n_clusters = min(n_zone, len(zone_df))
                if n_clusters > 1:
                    zone_indices = zone_df.index.to_numpy()
                    # Map index → position in self.corpus_ for embeddings lookup
                    # corpus_ indices are 0..n_corpus_-1; zone_df indices are aligned
                    zone_embeddings = self.embeddings[zone_indices]
                    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
                    zone_df = zone_df.copy()
                    zone_df["cluster"] = kmeans.fit_predict(zone_embeddings)

                    for c in range(n_clusters):
                        cluster_items = zone_df[zone_df["cluster"] == c]
                        if len(cluster_items) > 0:
                            selected.append(cluster_items.sample(1).index[0])
                else:
                    selected.extend(zone_df.sample(min(n_zone, len(zone_df))).index)
            else:
                selected.extend(zone_df.sample(min(n_zone, len(zone_df))).index)

        return df.loc[selected[:n_samples]]

    def fit(self, query: str):
        """Run the active-learning loop and train the classifier.

        Args:
            query: Search query/topic

        Returns:
            self
        """
        start_time = time.time()
        query = query.strip()[:15000]

        logger.info("Starting classification for query: %s", query)

        df = self.corpus_.copy()
        corpus_embeddings = self.embeddings  # triggers computation if needed

        non_empty = df[df["text"].str.strip() != ""]["text"]
        format_examples = non_empty.sample(min(_N_FORMAT_EXAMPLES, len(non_empty))).tolist()

        logger.info("Generating %s synthetic texts...", self.n_synthetic)
        synthetic_texts = generate_synthetic_texts(
            query, self.n_synthetic, format_examples, self.generation_prompt
        )
        logger.info("Generated %s synthetic texts", len(synthetic_texts))
        logger.info(synthetic_texts)

        logger.info("Embedding synthetic texts...")
        synthetic_embeddings = send_embedding_request(synthetic_texts)
        synthetic_embeddings = np.array(synthetic_embeddings)
        logger.info("Embedded %s synthetic texts", len(synthetic_embeddings))

        df["gt"] = pd.NA
        df["is_synthetic"] = False

        synthetic_df = pd.DataFrame({
            "text": synthetic_texts,
            "is_synthetic": True,
            "gt": True
        })
        synthetic_df.index = range(-1, -1 - len(synthetic_df), -1)

        # Combine both dataframes
        df = pd.concat([synthetic_df, df])

        # Compute distance to synthetic centroid
        all_embeddings = np.vstack([synthetic_embeddings, corpus_embeddings])
        synthetic_centroid = np.mean(synthetic_embeddings, axis=0).reshape(1, -1)
        synthetic_centroid = synthetic_centroid / np.linalg.norm(synthetic_centroid)
        df["cosine_similarity"] = all_embeddings @ synthetic_centroid.T

        # A priori probabilities are linear and evenly distributed on [0, 1] based on the cosine_similarity order
        df["prediction"] = df['cosine_similarity'].rank() / len(df)
        df["prediction_binary"] = df["prediction"] > 0.5
        df["confidence"] = (df["prediction"] - 0.5).abs()

        not_is_synthetic = ~df["is_synthetic"]

        logger.info("Selecting %s candidates for initial evaluation...", self.n_eval_per_iteration)
        candidates = self._select_candidates(df.loc[not_is_synthetic])

        # Active Learning loop
        logger.info("Starting active learning loop...")
        classifier = None
        prev_predictions = df.loc[not_is_synthetic, 'prediction'].values
        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            # Evaluate candidates with LLM
            logger.info("Iteration %s: Evaluating %s texts...", iteration, len(candidates))
            evaluations = evaluate(query=query, texts=candidates["text"].tolist(), prompt=self.evaluation_prompt)
            df.loc[candidates.index, "gt"] = pd.array(evaluations, dtype="boolean")

            yes_count = df.loc[candidates.index].query("gt == True").shape[0]
            no_count = df.loc[candidates.index].query("gt == False").shape[0]
            logger.info("Iteration %s: Evaluated %s texts. Yes: %s, No: %s", iteration, len(candidates), yes_count, no_count)

            # Train on labeled data
            labeled_mask = df["gt"].notna()
            X_train = all_embeddings[labeled_mask]
            y_train = df.loc[labeled_mask, "gt"].astype(bool).values

            # Check we have both classes
            if len(np.unique(y_train)) < 2:
                # Add farthest unlabeled as negatives
                unlabeled = df[not_is_synthetic & df["gt"].isna()]
                if len(unlabeled) > 0:
                    n_add = max(1, int(y_train.sum()))
                    farthest = unlabeled.nsmallest(n_add, "cosine_similarity")
                    df.loc[farthest.index, "gt"] = False
                    labeled_mask = df["gt"].notna()
                    X_train = all_embeddings[labeled_mask]
                    y_train = df.loc[labeled_mask, "gt"].astype(bool).values
                    logger.info("Added %s distant texts as negatives", len(farthest))

            if len(np.unique(y_train)) < 2:
                logger.info("Iteration %s: Skipping - need both classes", iteration)
                continue

            # Train classifier
            dist_dict = df.loc[labeled_mask, "gt"].value_counts().to_dict()
            logger.info("Iteration %s: Training classifier on %s texts (%s)...", iteration, len(y_train), dist_dict)
            classifier = LogisticRegressionClassifier(c=self.c_value)
            classifier.fit(X_train, y_train)
            logger.info("Iteration %s: Trained classifier on %s texts (%s).", iteration, len(y_train), dist_dict)

            # Predict
            logger.info("Iteration %s: Predicting all texts with freshly trained classifier...", iteration)
            df["prediction"] = classifier.predict_proba(all_embeddings)[:, -1]
            df["prediction_binary"] = df["prediction"] > 0.5
            df["confidence"] = (df["prediction"] - 0.5).abs()
            pred_dict = df.loc[not_is_synthetic, "prediction_binary"].value_counts().to_dict()
            logger.info("Iteration %s: Predicted all texts: %s", iteration, pred_dict)

            # Positive RMSE
            predictions = df.loc[not_is_synthetic, 'prediction'].values
            positive_zone = (
                (prev_predictions >= 0.5) | (predictions >= 0.5)
            )
            positive_zone_rmse = np.sqrt(np.mean((predictions[positive_zone] - prev_predictions[positive_zone]) ** 2)) if positive_zone.sum() > 0 else 0.0
            prev_predictions = predictions.copy()
            logger.info("Positive zone RMSE: %.4f", positive_zone_rmse)

            # Early stopping
            if iteration > self.min_iterations and positive_zone_rmse < self.positive_zone_rmse_threshold:
                logger.info("Early stop: Positive zone RMSE (%.4f) below threshold (%.4f).", positive_zone_rmse, self.positive_zone_rmse_threshold)
                break

            # Select next candidates
            if iteration < self.max_iterations:
                logger.info("Iteration %s: Selecting candidates for next iteration...", iteration)
                candidates = self._select_candidates(df.loc[not_is_synthetic & df["gt"].isna()])
                if len(candidates) == 0:
                    logger.info("No candidates found, stopping.")
                    break
                logger.info("Iteration %s: Selected %s candidates to evaluate...", iteration, len(candidates))

        self.classifier_ = classifier
        self.query_ = query
        self.iterations_ = iteration
        self.execution_time_ = time.time() - start_time
        logger.info("Done! Time: %.2fs", self.execution_time_)
        return self

    def predict_proba(self, df: pd.DataFrame | None = None) -> np.ndarray:
        """Return the probability that each text matches the query.

        Args:
            df: DataFrame with a ``text`` column. If *None*, uses the corpus
                supplied at initialisation. Embeddings are generated
                automatically if absent.

        Returns:
            1-D array of shape (n_samples,) with P(positive) for each text.
        """
        if self.classifier_ is None:
            raise ValueError("This ActiveLearner instance is not fitted yet. Call 'fit' before predicting.")

        if df is None:
            return self.classifier_.predict_proba(self.embeddings)[:, 1]

        embeddings = np.array(send_embedding_request(df["text"].tolist()))
        return self.classifier_.predict_proba(embeddings)[:, 1]

    def predict(self, df: pd.DataFrame | None = None) -> np.ndarray:
        """Return binary predictions for each text.

        Args:
            df: DataFrame with a ``text`` column. If *None*, uses the corpus
                supplied at initialisation. Embeddings are generated
                automatically if absent.

        Returns:
            Boolean array of shape (n_samples,) with True for predicted positives.
        """
        return self.predict_proba(df) >= 0.5
