import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from .embedding_service import create_embedding_service

# Load environment variables from .env file at project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class DatasetBuilder:
    """Build and save datasets with embeddings."""
    
    def __init__(self, embedding_model: str, api_key: str = None):
        """
        Initialize dataset builder with parameters.
        
        Args:
            embedding_model: Name of embedding model (e.g., "text-embedding-3-small")
            api_key: API key (if None, loads from OPENAI_API_KEY environment variable)
        """
        # Load API key from environment variable if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key not found. Please set OPENAI_API_KEY environment variable "
                    "or create a .env file at the project root with OPENAI_API_KEY=your-key"
                )
        
        self.embedding_service = create_embedding_service(api_key, embedding_model)
        self.embedding_model = embedding_model
        
    def build_dataset(self, df: pd.DataFrame, text_column: str, dataset_path: str, 
                     batch_size: int = 256):
        """
        Build dataset from dataframe and save to disk.
        
        Args:
            df: Input dataframe
            text_column: Name of column containing text to embed
            dataset_path: Path to save dataset (directory will be created)
            batch_size: Batch size for embedding generation
            
        Returns:
            Path to created dataset
        """
        # Create dataset directory
        os.makedirs(dataset_path, exist_ok=True)
        
        # Extract text column
        texts = df[text_column].tolist()
        print(f"Generating embeddings for {len(texts)} texts...")
        
        # Generate embeddings
        embeddings = self.embedding_service.embed_texts(texts, batch_size=batch_size)
        embeddings_array = np.array(embeddings)
        
        print(f"Generated embeddings with shape: {embeddings_array.shape}")
        
        # Save dataframe
        df_path = os.path.join(dataset_path, "dataframe.parquet")
        df.to_parquet(df_path, index=False)
        print(f"Saved dataframe to {df_path}")
        
        # Save embeddings
        embeddings_path = os.path.join(dataset_path, "embeddings.npy")
        np.save(embeddings_path, embeddings_array)
        print(f"Saved embeddings to {embeddings_path}")
        
        # Create metadata
        metadata = {
            "embedding_service": self.embedding_service.get_model_info(),
            "dataframe_info": {
                "shape": list(df.shape),
                "columns": df.columns.tolist(),
                "text_column": text_column
            },
            "created_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(dataset_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
        
        return dataset_path
    
    def build_dataset_from_existing_embeddings(self, df: pd.DataFrame, embeddings: np.ndarray, 
                                               text_column: str, dataset_path: str):
        """
        Build dataset from existing embeddings numpy array.
        
        Args:
            df: Input dataframe
            embeddings: Numpy array of embeddings where first dimension matches dataframe rows
            text_column: Name of column containing text (for metadata only)
            dataset_path: Path to save dataset (directory will be created)
            
        Returns:
            Path to created dataset
            
        Raises:
            ValueError: If embeddings first dimension doesn't match dataframe size
        """
        # Validate embeddings shape - only check number of rows
        n_rows = len(df)
        if embeddings.shape[0] != n_rows:
            raise ValueError(
                f"Embeddings array first dimension ({embeddings.shape[0]}) doesn't match "
                f"dataframe size ({n_rows}). Expected first dimension: {n_rows}"
            )
        
        print(f"Using existing embeddings with shape: {embeddings.shape}")
        print(f"Embeddings match dataframe size: {n_rows} rows")
        
        # Create dataset directory
        os.makedirs(dataset_path, exist_ok=True)
        
        # Save dataframe
        df_path = os.path.join(dataset_path, "dataframe.parquet")
        df.to_parquet(df_path, index=False)
        print(f"Saved dataframe to {df_path}")
        
        # Save embeddings
        embeddings_path = os.path.join(dataset_path, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path}")
        
        # Create metadata
        metadata = {
            "embedding_service": self.embedding_service.get_model_info(),
            "dataframe_info": {
                "shape": list(df.shape),
                "columns": df.columns.tolist(),
                "text_column": text_column
            },
            "embeddings_shape": list(embeddings.shape),
            "created_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(dataset_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
        
        return dataset_path
