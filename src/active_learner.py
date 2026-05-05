import os
import json
import time
import numpy as np
import pandas as pd
from uuid import uuid4
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import MiniBatchKMeans
from dotenv import load_dotenv

from alira.classifiers import LogisticRegressionClassifier
from alira.llms import generate_documents, evaluate_documents

from alira.opensearch import fetch_all, embed

# Load environment variables from .env file at project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def select_stratified_diverse(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Stratified by confidence + diverse within each stratum."""
    if len(df) == 0 or n_samples <= 0:
        return df.head(0)

    df = df.copy()

    # Allocate budget across zones
    zones = [
        (df[df["prediction"] > 0.7], 0.4),  # 40% high confidence positive
        (df[df["prediction"].between(0.3, 0.7)], 0.4),  # 40% uncertain
        (df[df["prediction"] < 0.3], 0.2),  # 20% likely negative
    ]

    selected = []
    for zone_df, fraction in zones:
        n_zone = max(1, int(n_samples * fraction))
        if len(zone_df) == 0:
            continue

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


class ActiveLearner:
    """Active learning classifier for document filtering."""

    def __init__(
        self,
        index_name: str,
        document_type: str,
        generation_llm_model: str = "gpt-4o-mini",
        evaluation_llm_model: str = "gpt-5.2",
        n_synthetic_documents: int = 10,
        n_nearest_start: int = 40,
        n_iterations: int = 15,
        n_eval_per_iteration: int = 20,
        c_value: float = 1.0,
    ):
        """
        Initialize active learner with dataset.
        
        Args:
            index_name: OpenSearch index where to fetch documents from.
            document_types: Type of the documents to include in active learner.
            generation_llm_model: Generation LLM model name
            evaluation_llm_model: Evaluation LLM model name
            n_synthetic_documents: Number of synthetic documents to generate
            n_nearest_start: Number of nearest docs for initial labeling
            n_iterations: Maximum number of active learning iterations
            n_eval_per_iteration: Number of docs to evaluate per iteration
            c_value: C parameter for LogisticRegression
        """
        
        # Store parameters
        self.index_name = index_name
        self.document_type = document_type
        self.n_synthetic_documents = n_synthetic_documents
        self.n_nearest_start = n_nearest_start
        self.n_iterations = n_iterations
        self.n_eval_per_iteration = n_eval_per_iteration
        self.c_value = c_value
        self.log_file = None

        # Fetch data
        self._fetch()

    def _fetch(self):
        # Fetch data from OpenSearch index
        self._log(f"Fetching documents with type {self.document_type}...")

        response = fetch_all(self.index_name, document_type=self.document_type)
        hits = response['hits']['hits']

        self._log(f"Fetched {len(hits)} documents with type {self.document_type}")

        # Store data
        self.df = pd.DataFrame([hit["_source"] for hit in hits])

    def _log(self, message: str):
        """Log message to console and file."""
        print(message)
        if self.log_file:
            self.log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{message}\n")
            self.log_file.flush()
        
    def classify(self, query: str, output_dir: str = "results"):
        """
        Run active learning classification.
        
        Args:
            query: Search query/topic (used directly)
            output_dir: Output directory for results
            
        Returns:
            results_df (positive items with scores), session_dir, params_dict
        """
        start_time = time.time()
        query = query.strip()[:15000]
        
        # Setup session
        session_id = str(uuid4())
        session_dir = os.path.join(output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Initialize log file
        log_path = os.path.join(session_dir, "run.log")
        self.log_file = open(log_path, "w")
        
        self._log(f"Starting classification for query: {query}")
        
        # Generate synthetic documents
        self._log(f"Generating {self.n_synthetic_documents} synthetic documents...")
        synthetic_documents = generate_documents(query, self.n_synthetic_documents, self.document_type)
        self._log(f"Generated {len(synthetic_documents)} synthetic documents")
        
        # Embed synthetic documents
        self._log("Embedding synthetic documents...")
        synthetic_embeddings = embed(synthetic_documents)
        synthetic_embeddings = np.array(synthetic_embeddings)
        self._log(f"Embedded {len(synthetic_embeddings)} synthetic documents")

        # Build documents dataframe
        documents_df = self.df[['text', 'embedding']].copy()
        documents_df["gt"] = pd.NA
        documents_df["is_synthetic"] = False

        # Build synthetic dataframe
        synthetic_df = pd.DataFrame({
            "text": synthetic_documents,
            "embedding": [synthetic_embeddings[i] for i in range(len(synthetic_documents))],
            "is_synthetic": True,
            "gt": True
        })
        synthetic_df.index = range(-1, -1 - len(synthetic_df), -1)
        
        # Combine both dataframes
        df = pd.concat([synthetic_df, documents_df])

        # Select initial candidates to evaluate as the closest to the synthetic centroid
        all_embeddings = np.vstack(df["embedding"].values)
        synthetic_centroid = np.mean(synthetic_embeddings, axis=0).reshape(1, -1)
        synthetic_centroid = synthetic_centroid / np.linalg.norm(synthetic_centroid)
        df["distance_to_centroid"] = euclidean_distances(all_embeddings, synthetic_centroid).ravel()

        not_is_synthetic = ~df["is_synthetic"]

        self._log(f"Selecting the closest {self.n_nearest_start} documents to the synthetic centroid as candidates to evaluate...")
        candidates = df.loc[not_is_synthetic].nsmallest(self.n_nearest_start, "distance_to_centroid")

        # Active Learning loop
        self._log("Starting active learning loop...")
        classifier = None
        prev_positives = None
        early_stop_threshold = 0.02
        for iteration in range(1, self.n_iterations + 1):
            # Evaluate candidates with LLM
            evaluations = evaluate_documents(topic=query, texts=candidates["text"].tolist())
            df.loc[candidates.index, "gt"] = pd.array(evaluations, dtype="boolean")

            yes_count = df.loc[candidates.index].query("gt == True").shape[0]
            no_count = df.loc[candidates.index].query("gt == False").shape[0]
            self._log(f"Iteration {iteration}: Evaluated {len(candidates)} documents. Yes: {yes_count}, No: {no_count}")

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
                    farthest = unlabeled.nlargest(n_add, "distance_to_centroid")
                    df.loc[farthest.index, "gt"] = False
                    training_df = df.dropna(subset=["gt"]).copy()
                    training_df["gt"] = training_df["gt"].astype(bool)
                    X_train = np.vstack(training_df["embedding"].values)
                    y_train = training_df["gt"].values
                    self._log(f"Added {len(farthest)} distant docs as negatives")
            
            if len(np.unique(y_train)) < 2:
                self._log(f"Iteration {iteration}: Skipping - need both classes")
                continue
            
            # Train classifier
            classifier = LogisticRegressionClassifier(c=self.c_value)
            classifier.fit(X_train, y_train)
            
            # Predict
            all_embeddings = np.vstack(df["embedding"].values)
            df["prediction"] = classifier.predict_proba(all_embeddings)[:, -1]
            df["prediction_binary"] = df["prediction"] > 0.5
            df["confidence"] = (df["prediction"] - 0.5).abs()
            
            dist_dict = training_df["gt"].value_counts().to_dict()
            pred_dict = df.loc[not_is_synthetic, "prediction_binary"].value_counts().to_dict()
            self._log(f"Iteration {iteration}: Trained with {dist_dict}. Predictions: {pred_dict}")
            
            # Early stopping
            positives = set(df.index[not_is_synthetic & df["prediction_binary"]])
            if prev_positives is not None:
                flipped = len(positives ^ prev_positives)
                total = len(positives | prev_positives)
                flip_rate = flipped / total if total > 0 else 0
                self._log(f"Flip rate: {flip_rate*100:.2f}%")
                if flip_rate < early_stop_threshold:
                    self._log(f"Early stop (flip-rate < {early_stop_threshold*100:.1f}%)")
                    break
            prev_positives = positives.copy()
            
            # Select next candidates
            unlabeled = df[not_is_synthetic & df["gt"].isna()]
            if len(unlabeled) == 0:
                self._log("All documents labeled, stopping.")
                break
            
            candidates = select_stratified_diverse(unlabeled, self.n_eval_per_iteration)
            if len(candidates) == 0:
                self._log("No candidates found, stopping.")
                break

        # Keep only real documents with a positive prediction
        positives = df.loc[not_is_synthetic & df['prediction_binary']]

        # Add score column
        results_df = self.df.loc[positives.index].copy()
        results_df["score"] = positives["prediction"]

        # Sort by score
        results_df = results_df.sort_values("score", ascending=False)
        
        # Save results
        results_path = os.path.join(session_dir, "results.parquet")
        results_df.to_parquet(results_path, index=False)
        self._log(f"Saved results to {results_path}")
        
        # Save model
        if classifier.model:
            model_path = os.path.join(session_dir, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(classifier.model, f)
            self._log(f"Saved model to {model_path}")
        
        # Save parameters
        elapsed = time.time() - start_time
        params = {
            "session_id": session_id,
            "index_name": self.index_name,
            "document_type": self.document_type,
            "query": query,
            "n_synthetic_documents": self.n_synthetic_documents,
            "n_nearest_start": self.n_nearest_start,
            "n_iterations": self.n_iterations,
            "n_eval_per_iteration": self.n_eval_per_iteration,
            "execution_times": {
                "total_seconds": elapsed
            },
            "statistics": {
                "total_items": len(self.df),
                "positive_items": len(results_df),
                "negative_items": len(self.df) - len(results_df),
                "iterations_completed": iteration
            },
            "model_info": {
                "type": "LogisticRegression",
                "c_value": self.c_value
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        params_path = os.path.join(session_dir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
        self._log(f"Saved parameters to {params_path}")
        
        # Close log file
        self.log_file.close()
        self.log_file = None
        
        self._log(f"Done! Time: {elapsed:.2f}s")
        
        return results_df, session_dir, params
