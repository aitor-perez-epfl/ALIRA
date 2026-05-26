"""Lab Explorer Twin

Script trying to replicate Lab Explorer using a simpler ALIRA version that doesn't handle the data
"""

from dotenv import load_dotenv

load_dotenv()

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import epfl_data_index as edi

from alira.active_learner import ActiveLearner

# Setup logging to file and console
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = Path(f"results/{timestamp}")
output_dir.mkdir(exist_ok=True)
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

doc_type = "professor"
query = "machine learning"

logger.info("Fetching documents with type `%s`...", doc_type)
response = edi.fetch_all(doc_type=doc_type)
hits = response['hits']['hits']
X = pd.DataFrame([hit["_source"] for hit in hits])
logger.info("Fetched %s documents with type %s", len(X), doc_type)

# Find publications related to the query
logger.info("Preparing Active Learner for documents with type `%s`...", doc_type)
learner = ActiveLearner()

logger.info("Running classification for query=`%s`...", query)
results_df, params = learner.fit(df=X, query=query)

# Save results to CSV
results_path = output_dir / "results.csv"
results_df.to_csv(results_path, index=False)
logger.info("Saved results to %s", results_path)

logger.info("\nFound %s positive items (`%s` about `%s`)", len(results_df), doc_type, query)
logger.info("Parameters: %s", params)
logger.info("\nFirst few results:")
for idx, row in results_df.head().iterrows():
    logger.info("  - %s...", row['text'][:80])
