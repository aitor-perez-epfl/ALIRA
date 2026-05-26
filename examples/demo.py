from dotenv import load_dotenv

load_dotenv()

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from alira import ActiveLearner

# Setup logging to file and console
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = Path(f"results/demo-{timestamp}")
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

# Read dataset with movie texts
# (built from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
logger.info("Loading dataset...")
movies = pd.read_csv("data/movies.csv")[:100]
logger.info("Loaded %s movies", len(movies))

learner = ActiveLearner()
logger.info("Starting classification for query: 'movies for children'")
children_movies, params = learner.fit(df=movies, query="movies for children")

# Save results to CSV
results_path = output_dir / "results.csv"
children_movies.to_csv(results_path, index=False)
logger.info("Saved results to %s", results_path)

logger.info("Done!")
