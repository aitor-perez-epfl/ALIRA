"""
Lab Explorer Twin

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

# Logs
log_path = Path("logs/lab_explorer") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

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

logger.info(f"Fetching documents with type `{doc_type}`...")
response = edi.fetch_all(doc_type=doc_type)
hits = response['hits']['hits']
X = pd.DataFrame([hit["_source"] for hit in hits])
logger.info(f"Fetched {len(X)} documents with type {doc_type}")

# Find publications related to the query
logger.info(f"Preparing Active Learner for documents with type `{doc_type}`...")
learner = ActiveLearner()

logger.info(f"Running classification for query=`{query}`...")
results_df, session_dir, params = learner.fit(df=X, query=query)

logger.info(f"\nFound {len(results_df)} positive items (`{doc_type}` about `{query}`)")
logger.info(f"Results saved to: {session_dir}")
logger.info(f"\nFirst few results:")
for idx, row in results_df.head().iterrows():
    logger.info(f"  - {row['text'][:80]}...")
