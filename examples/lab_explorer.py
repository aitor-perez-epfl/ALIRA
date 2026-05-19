"""
Lab Explorer Twin

Script trying to replicate Lab Explorer using a simpler ALIRA version that doesn't handle the data
"""

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import epfl_data_index as edi
from alira.active_learner import ActiveLearner

doc_type = "professor"
query = "machine learning"

import logging

logger = logging.getLogger(__name__)

fh = logging.FileHandler('your/path')
logger.addHandler(fh)

logger.info(f"Fetching documents with type `{doc_type}`...")
response = edi.fetch_all(doc_type=doc_type)
hits = response['hits']['hits']
X = pd.DataFrame([hit["_source"] for hit in hits])
print(f"Fetched {len(X)} documents with type {doc_type}")

# Find publications related to the query
print(f"Preparing Active Learner for documents with type `{doc_type}`...")
learner = ActiveLearner()

print(f"Running classification for query=`{query}`...")
learner.fit(query=query, X=X)

y = learner.predict_proba(X)

results_df, session_dir, params = learner.classify(query=query)

print(f"\nFound {len(results_df)} positive items (`{document_type}` about `{query}`)")
print(f"Results saved to: {session_dir}")
print(f"\nFirst few results:")
for idx, row in results_df.head().iterrows():
    print(f"  - {row['text'][:80]}...")
