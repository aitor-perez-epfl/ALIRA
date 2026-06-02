from dotenv import load_dotenv

load_dotenv()

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from alira import ActiveLearner

# Setup logging to file and console
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = Path(f"results/ignobel-{timestamp}")
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

logger = logging.getLogger(__name__)

################################################################

# Read dataset
logger.info("Loading dataset...")
df = pd.read_csv("data/ig_nobel_candidates.csv")
logger.info("Loaded %s rows", len(df))

learner = ActiveLearner(corpus=df["title"])
query = "sad and depressing publications"
logger.info("Starting classification for query: %s", query)
learner.fit(query=query)

# Get predictions
df["score"] = learner.predict_proba()
results_df = df[df["score"] >= 0.5].sort_values("score", ascending=False)

# Save results to CSV
results_path = output_dir / "results.csv"
results_df.to_csv(results_path, index=False)
logger.info("Saved results to %s", results_path)

logger.info("Done!")
