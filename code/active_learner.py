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

from .embedding_service import create_embedding_service
from .generation_llm import create_generation_llm
from .evaluation_llm import create_evaluation_llm
from .classifiers import LogisticRegressionClassifier

# Load environment variables from .env file at project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


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
    
    def __init__(self, dataset_path: str, 
                 embedding_model: str = None,
                 generation_llm_model: str = "gpt-4o-mini",
                 evaluation_llm_model: str = "gpt-5.2",
                 api_key: str = None,
                 n_synthetic_titles: int = 10,
                 n_nearest_start: int = 40,
                 n_iterations: int = 15,
                 n_eval_per_iteration: int = 20,
                 evaluation_query: str = None,
                 c_value: float = 1.0):
        """
        Initialize active learner with dataset.
        
        Args:
            dataset_path: Path to dataset directory
            embedding_model: Embedding model name (must match dataset, auto-detected if None)
            generation_llm_model: Generation LLM model name
            evaluation_llm_model: Evaluation LLM model name
            api_key: API key (if None, loads from OPENAI_API_KEY environment variable)
            n_synthetic_titles: Number of synthetic titles to generate
            n_nearest_start: Number of nearest docs for initial labeling
            n_iterations: Maximum number of active learning iterations
            n_eval_per_iteration: Number of docs to evaluate per iteration
            evaluation_query: Custom evaluation query template
            c_value: C parameter for LogisticRegression
        """
        # Load API key from environment variable if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key not found. Please set OPENAI_API_KEY environment variable "
                    "or create a .env file at the project root with OPENAI_API_KEY=your-key"
                )
        
        # Load dataset metadata
        with open(f"{dataset_path}/metadata.json") as f:
            self.metadata = json.load(f)
        
        # Load dataframe and embeddings
        self.df = pd.read_parquet(f"{dataset_path}/dataframe.parquet")
        self.embeddings = np.load(f"{dataset_path}/embeddings.npy")
        
        # Validate embeddings shape
        if len(self.embeddings) != len(self.df):
            raise ValueError(f"Embeddings shape {len(self.embeddings)} doesn't match dataframe length {len(self.df)}")
        
        # Determine embedding model (from param or dataset metadata)
        if embedding_model is None:
            embedding_model = self.metadata["embedding_service"]["model"]
        
        # Create services using factories
        self.embedding_service = create_embedding_service(api_key, embedding_model)
        self.generation_llm = create_generation_llm(api_key, generation_llm_model)
        self.evaluation_llm = create_evaluation_llm(api_key, evaluation_llm_model)
        
        # Validate embedding service matches dataset
        self._validate_embedding_service()
        
        # Store parameters
        self.dataset_path = dataset_path
        self.n_synthetic_titles = n_synthetic_titles
        self.n_nearest_start = n_nearest_start
        self.n_iterations = n_iterations
        self.n_eval_per_iteration = n_eval_per_iteration
        self.evaluation_query = evaluation_query or "Classify each paper as related (1) or not related (0) to: {topic}"
        self.c_value = c_value
        self.log_file = None
        
    def _validate_embedding_service(self):
        """Validate that current embedding service matches dataset's embedding service."""
        dataset_embedding_info = self.metadata["embedding_service"]
        current_embedding_info = self.embedding_service.get_model_info()
        
        if dataset_embedding_info != current_embedding_info:
            raise ValueError(
                f"Embedding service mismatch! "
                f"Dataset was created with {dataset_embedding_info}, "
                f"but current service is {current_embedding_info}"
            )
        
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
        
        # Get text column from metadata
        text_column = self.metadata["dataframe_info"]["text_column"]
        
        # Generate synthetic titles
        self._log(f"Generating {self.n_synthetic_titles} synthetic titles...")
        titles = None
        for attempt in range(3):
            titles = self.generation_llm.generate_titles(query, self.n_synthetic_titles)
            if titles:
                break
        if not titles:
            raise ValueError("Failed to generate titles after 3 attempts")
        
        self._log(f"Generated {len(titles)} synthetic titles")
        
        # Embed synthetic titles
        self._log("Embedding synthetic titles...")
        synthetic_emb = self.embedding_service.embed_texts(titles)
        synthetic_emb_array = np.array(synthetic_emb)
        
        # Build working dataframe
        working_df = self.df.copy()
        working_df["embedding"] = [self.embeddings[i] for i in range(len(working_df))]
        working_df["gt"] = pd.NA
        working_df["is_synthetic"] = False
        
        # Create synthetic dataframe
        synthetic_df = pd.DataFrame({
            text_column: titles,
            "embedding": [synthetic_emb_array[i] for i in range(len(titles))],
            "is_synthetic": True,
            "gt": True
        })
        synthetic_df.index = range(-1, -1 - len(synthetic_df), -1)
        
        # Combine
        ldf = pd.concat([synthetic_df, working_df])
        
        # Distance to centroid
        centroid = np.mean(synthetic_emb_array, axis=0).reshape(1, -1)
        all_embeddings = np.vstack(ldf["embedding"].values)
        ldf["distance_to_centroid"] = euclidean_distances(all_embeddings, centroid).ravel()
        
        real_docs_mask = ~ldf["is_synthetic"]
        
        # Initial labeling: N nearest docs
        self._log(f"Selecting {self.n_nearest_start} nearest documents for initial labeling...")
        closest = ldf.loc[real_docs_mask].nsmallest(self.n_nearest_start, "distance_to_centroid")
        
        evaluation = self.evaluation_llm.evaluate(
            closest[text_column].tolist(),
            query,
            self.evaluation_query
        )
        if evaluation is None:
            raise ValueError("LLM evaluation failed after all retries")
        
        ldf.loc[closest.index, "gt"] = pd.array(evaluation, dtype="boolean")
        
        validated_count = ldf.loc[closest.index].query("gt == True").shape[0]
        rejected_count = ldf.loc[closest.index].query("gt == False").shape[0]
        self._log(f"Initial labeling: {validated_count} validated, {rejected_count} rejected")
        
        # Active learning loop
        self._log("Starting active learning iterations...")
        model = None
        classifier = None
        prev_pos_idx = None
        early_stop_threshold = 0.02
        iteration = 0
        
        for iteration in range(1, self.n_iterations + 1):
            # Train on labeled data
            training_df = ldf.dropna(subset=["gt"]).copy()
            training_df["gt"] = training_df["gt"].astype(bool)
            
            X_train = np.vstack(training_df["embedding"].values)
            y_train = training_df["gt"].values
            
            # Check we have both classes
            if len(np.unique(y_train)) < 2:
                # Add farthest unlabeled as negatives
                unlabeled = ldf[real_docs_mask & ldf["gt"].isna()]
                if len(unlabeled) > 0:
                    n_add = max(1, int(y_train.sum()))
                    farthest = unlabeled.nlargest(n_add, "distance_to_centroid")
                    ldf.loc[farthest.index, "gt"] = False
                    training_df = ldf.dropna(subset=["gt"]).copy()
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
            model = classifier.model
            
            # Predict
            all_embeddings = np.vstack(ldf["embedding"].values)
            ldf["prediction"] = classifier.predict_proba(all_embeddings)[:, -1]
            ldf["prediction_binary"] = ldf["prediction"] > 0.5
            ldf["uncertainty"] = (ldf["prediction"] - 0.5).abs()
            
            dist_dict = training_df["gt"].value_counts().to_dict()
            pred_dict = ldf.loc[real_docs_mask, "prediction_binary"].value_counts().to_dict()
            self._log(f"Iteration {iteration}: Trained with {dist_dict}. Predictions: {pred_dict}")
            
            if iteration == self.n_iterations:
                break
            
            # Early stopping
            curr_pos_idx = set(ldf.index[real_docs_mask & ldf["prediction_binary"]])
            if prev_pos_idx is not None:
                flipped = curr_pos_idx.symmetric_difference(prev_pos_idx)
                denom = len(curr_pos_idx | prev_pos_idx)
                flip_rate = len(flipped) / denom if denom > 0 else 0.0
                self._log(f"Flip rate: {flip_rate*100:.2f}%")
                if flip_rate < early_stop_threshold:
                    self._log(f"Early stop (flip-rate < {early_stop_threshold*100:.1f}%)")
                    break
            prev_pos_idx = curr_pos_idx.copy()
            
            # Select next batch
            unlabeled = ldf[real_docs_mask & ldf["gt"].isna()]
            if len(unlabeled) == 0:
                self._log("All documents labeled.")
                break
            
            to_evaluate = select_stratified_diverse(unlabeled, self.n_eval_per_iteration)
            if len(to_evaluate) == 0:
                break
            
            eval_results = self.evaluation_llm.evaluate(
                to_evaluate[text_column].tolist(),
                query,
                self.evaluation_query
            )
            if eval_results is None:
                self._log("LLM evaluation failed, stopping.")
                break
            
            ldf.loc[to_evaluate.index, "gt"] = pd.array(eval_results, dtype="boolean")
            
            yes_count = ldf.loc[to_evaluate.index].query("gt == True").shape[0]
            no_count = ldf.loc[to_evaluate.index].query("gt == False").shape[0]
            self._log(f"Iteration {iteration}: Labeled {len(to_evaluate)}. Yes: {yes_count}, No: {no_count}")
        
        # Final predictions (use last trained classifier)
        if classifier is None:
            # Train one final classifier if we don't have one
            training_df = ldf.dropna(subset=["gt"]).copy()
            if len(training_df) > 0:
                training_df["gt"] = training_df["gt"].astype(bool)
                X_train = np.vstack(training_df["embedding"].values)
                y_train = training_df["gt"].values
                if len(np.unique(y_train)) >= 2:
                    classifier = LogisticRegressionClassifier(c=self.c_value)
                    classifier.fit(X_train, y_train)
                    model = classifier.model
        
        if classifier is None:
            raise ValueError("Could not train classifier - insufficient labeled data")
        
        all_embeddings = np.vstack(ldf["embedding"].values)
        ldf["prediction"] = classifier.predict_proba(all_embeddings)[:, -1]
        ldf["prediction_binary"] = ldf["prediction"] > 0.5
        
        # Filter for positive items
        real_results = ldf.loc[real_docs_mask].copy()
        positive_results = real_results[real_results["prediction_binary"] == True].copy()
        
        # Add score column
        results_df = self.df.loc[positive_results.index].copy()
        results_df["score"] = positive_results["prediction"].values
        
        # Sort by score
        results_df = results_df.sort_values("score", ascending=False)
        
        # Save results
        results_path = os.path.join(session_dir, "results.parquet")
        results_df.to_parquet(results_path, index=False)
        self._log(f"Saved results to {results_path}")
        
        # Save model
        if model:
            model_path = os.path.join(session_dir, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            self._log(f"Saved model to {model_path}")
        
        # Save parameters
        elapsed = time.time() - start_time
        params = {
            "session_id": session_id,
            "dataset_path": self.dataset_path,
            "query": query,
            "text_column": text_column,
            "embedding_model": self.embedding_service.get_model_info(),
            "generation_llm_model": {"provider": "openai", "model": self.generation_llm.model},
            "evaluation_llm_model": {"provider": "openai", "model": self.evaluation_llm.model},
            "n_synthetic_titles": self.n_synthetic_titles,
            "n_nearest_start": self.n_nearest_start,
            "n_iterations": self.n_iterations,
            "n_eval_per_iteration": self.n_eval_per_iteration,
            "evaluation_query": self.evaluation_query,
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
