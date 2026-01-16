"""
ALIRA – Active Learning Iterative Retrieval Agent

Simple example showing how to use ALIRA.
This example uses a CSV file with 20 scientific papers (10 about robotics, 10 about deep learning).
"""

import pandas as pd
from code.dataset_builder import DatasetBuilder
from code.active_learner import ActiveLearner


# Step 1: Load the example CSV file
df = pd.read_csv("example/papers.csv")

# Step 2: Build the dataset (this will create embeddings)
builder = DatasetBuilder(
    embedding_model="text-embedding-3-small"
)

dataset_path = builder.build_dataset(
    df=df,
    text_column="text",
    dataset_path="datasets/papers_dataset"
)
print(f"Dataset saved to: {dataset_path}\n")

# Step 3: Run classification
# Try searching for "robotics" to find robotics-related papers
print("Running classification for 'robotics' papers...")
learner = ActiveLearner(
    dataset_path="datasets/papers_dataset"
)

results_df, session_dir, params = learner.classify(query="robotics")

print(f"\nFound {len(results_df)} positive items (robotics papers)")
print(f"Results saved to: {session_dir}")
print(f"\nFirst few results:")
for idx, row in results_df.head().iterrows():
    print(f"  - {row['text'][:80]}...")
