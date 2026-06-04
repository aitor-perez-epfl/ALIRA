"""Active learning demo with sample datasets."""

from dotenv import load_dotenv

load_dotenv()

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alira import ActiveLearner

# Dataset and query
dataset = "movies"
query = "movies about parenthood"

# Data paths
csv_path = f"data/{dataset}.csv"
embeddings_path = f"data/{dataset}_embeddings.npy"

################################################################

# Logging
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = Path(f"results/{dataset}-{timestamp}")
output_dir.mkdir(exist_ok=True, parents=True)
log_path = output_dir / "run.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

################################################################

# Load data and embeddings
logger.info("Loading dataset from %s...", csv_path)
df = pd.read_csv(csv_path)
logger.info("Loaded %s rows", len(df))

embedding_path = Path(embeddings_path)
if embedding_path.exists():
    logger.info("Loading cached embeddings...")
    embeddings = np.load(embedding_path)
    logger.info("Loaded embeddings with shape %s", embeddings.shape)
else:
    logger.info("No cached embeddings found, will compute on demand.")
    embeddings = None

################################################################

# Active learning classifier
learner = ActiveLearner(corpus=df["text"], embeddings=embeddings)
logger.info("Starting classification for query: %s", query)
learner.fit(query=query)

# Get predictions
df["score"] = learner.predict_proba()
results_df = df[df["score"] >= 0.5].sort_values("score", ascending=False)

# Save results
results_path = output_dir / "results.csv"
results_df.to_csv(results_path, index=False)
logger.info("Saved results to %s", results_path)

logger.info("Done!")
