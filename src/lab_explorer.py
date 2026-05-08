"""
Lab Explorer Twin

Example replica of Lab Explorer using a simpler ALIRA version where data and embeddings come from OpenSearch.
"""

import numpy as np
import pandas as pd

from alira.active_learner import ActiveLearner

from alira.opensearch import search

index_name = "test3"
document_type = "publication"
query = "machine learning"

# Step 2: Find publications related to the query
print(f"Preparing Active Learner for documents with type `{document_type}`...")
learner = ActiveLearner(
    index_name=index_name,
    document_type=document_type
)

print(f"Running classification for query=`{query}`...")
results_df, session_dir, params = learner.classify(query=query)

print(f"\nFound {len(results_df)} positive items (`{document_type}` about `{query}`)")
print(f"Results saved to: {session_dir}")
print(f"\nFirst few results:")
for idx, row in results_df.head().iterrows():
    print(f"  - {row['text'][:80]}...")
