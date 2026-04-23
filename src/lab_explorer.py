"""
Lab Explorer Twin

Example replica of Lab Explorer using a simpler ALIRA version where data and embeddings come from OpenSearch.
"""

import numpy as np
import pandas as pd

from alira.active_learner import ActiveLearner

from alira.opensearch import search

index_name = "test2"
document_type = "publication"
query = "robotics"

# Step 2: Find publications related to robotics
print(f"Running classification for `{document_type}` related to `{query}`...")
learner = ActiveLearner(
    index_name=index_name,
    document_type=document_type
)

results_df, session_dir, params = learner.classify(query=query)

print(f"\nFound {len(results_df)} positive items (robotics papers)")
print(f"Results saved to: {session_dir}")
print(f"\nFirst few results:")
for idx, row in results_df.head().iterrows():
    print(f"  - {row['text'][:80]}...")
