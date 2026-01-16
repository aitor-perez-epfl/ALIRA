# ALIRA – Active Learning Iterative Retrieval Agent

Combines RAG with active learning to iteratively discover relevant documents from large corpora using LLM validation and classifier refinement.

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file at the project root with your API key:

```bash
OPENAI_API_KEY=sk-...
```

You can copy the example file:
```bash
cp .env.example .env
```

Then edit `.env` and add your actual API key. The `.env` file is automatically ignored by git to keep your API key secure.

## Quick Start

### 1. Build the Dataset

```python
import pandas as pd
from code.dataset_builder import DatasetBuilder

# Load your dataframe
df = pd.read_parquet("your_data.parquet")

# Build dataset with embeddings
builder = DatasetBuilder(
    embedding_model="text-embedding-3-small"
)

dataset_path = builder.build_dataset(
    df=df,
    text_column="text",  # Column name containing text to embed
    dataset_path="datasets/my_dataset"
)
```

This creates a dataset folder containing:
- `dataframe.parquet` - Your original dataframe
- `embeddings.npy` - Generated embeddings
- `metadata.json` - Dataset metadata

### 2. Run Classification

```python
from active_learner import ActiveLearner

# Initialize learner
learner = ActiveLearner(
    dataset_path="datasets/my_dataset"
)

# Classify documents
results_df, session_dir, params = learner.classify(query="machine learning")

print(f"Found {len(results_df)} positive items")
print(f"Results saved to: {session_dir}")
```

The results dataframe contains only positive items (documents matching the query) with confidence scores.

## Example

See `main.py` for a complete working example.

## What Happens

1. **Dataset Preparation**: Builds embeddings for your dataframe and saves as a reusable dataset
2. **Classification**: Uses active learning to iteratively identify documents matching your query
3. **Results**: Returns filtered dataframe with only positive items and their scores

Each classification run saves results to `results/{session_id}/` containing:
- `results.parquet` - Filtered positive items with scores
- `model.pkl` - Trained classifier model
- `params.json` - Parameters and execution statistics
- `run.log` - Progress log

## Advanced Usage

For advanced features like:
- Loading existing embeddings
- Customizing active learning parameters
- Changing evaluation queries
- Using different models

See `advanced_usage.md`.
