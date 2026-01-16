# Advanced Usage Guide

This guide covers advanced features and customization options for ALIRA (Active Learning Iterative Retrieval Agent).

## Loading Existing Embeddings

If you already have embeddings as a numpy array, you can build a dataset from them:

```python
import numpy as np
import pandas as pd
from code.dataset_builder import DatasetBuilder

# Load your dataframe and existing embeddings
df = pd.read_parquet("your_data.parquet")
embeddings = np.load("your_embeddings.npy")  # First dimension must match dataframe rows

# Build dataset from existing embeddings
builder = DatasetBuilder(
    embedding_model="text-embedding-3-small"  # Must match your embeddings model
)

dataset_path = builder.build_dataset_from_existing_embeddings(
    df=df,
    embeddings=embeddings,  # Numpy array where first dimension matches dataframe rows
    text_column="text",  # Column name (for metadata only)
    dataset_path="datasets/my_dataset"
)
```

**Important**: The embeddings array's first dimension must match the number of rows in the dataframe. The method will validate this and raise an error if sizes don't match.

## Customizing Active Learning Parameters

You can customize various parameters when initializing `ActiveLearner`:

```python
from code.active_learner import ActiveLearner

learner = ActiveLearner(
    dataset_path="datasets/my_dataset",
    embedding_model="text-embedding-3-small",  # Optional: auto-detected if None
    generation_llm_model="gpt-4o-mini",  # Model for synthetic title generation
    evaluation_llm_model="gpt-5.2",  # Model for document evaluation
    n_synthetic_titles=10,  # Number of synthetic titles to generate
    n_nearest_start=40,  # Number of nearest docs for initial labeling
    n_iterations=15,  # Maximum active learning iterations
    n_eval_per_iteration=20,  # Documents to evaluate per iteration
    c_value=1.0  # C parameter for LogisticRegression
)
```

### Parameter Descriptions

- `embedding_model`: Embedding model name (optional, auto-detected from dataset metadata)
- `generation_llm_model`: Model used to generate synthetic example titles (default: "gpt-4o-mini")
- `evaluation_llm_model`: Model used to evaluate documents (default: "gpt-5.2")
- `n_synthetic_titles`: Number of synthetic titles to generate as examples (default: 10)
- `n_nearest_start`: Number of nearest documents to evaluate initially (default: 40)
- `n_iterations`: Maximum number of active learning iterations (default: 15)
- `n_eval_per_iteration`: Number of documents to evaluate in each iteration (default: 20)
- `c_value`: Regularization parameter for LogisticRegression (default: 1.0)

## Custom Evaluation Query

You can customize the evaluation prompt template:

```python
learner = ActiveLearner(
    dataset_path="datasets/my_dataset",
    evaluation_query="Is this document about {topic}? Answer 1 for yes, 0 for no."
)
```

The `{topic}` placeholder will be replaced with your query string.

## Using Explicit API Keys

You can pass the API key directly instead of using environment variables:

```python
builder = DatasetBuilder(
    embedding_model="text-embedding-3-small",
    api_key="sk-..."  # Direct API key
)

learner = ActiveLearner(
    dataset_path="datasets/my_dataset",
    api_key="sk-..."  # Direct API key
)
```

## Supported Models

### Embedding Models
- `text-embedding-3-small`
- `text-embedding-3-large`
- `text-embedding-ada-002`

### Generation LLM Models (for synthetic titles)
- `gpt-4o-mini`
- `gpt-3.5-turbo`
- `gpt-4`
- `gpt-5-mini`

### Evaluation LLM Models (for document evaluation)
- `gpt-5.2`
- `gpt-4`
- `gpt-4-turbo`

## How Active Learning Works

1. **Synthetic Title Generation**: Generates example titles related to your query using the generation LLM
2. **Initial Labeling**: Selects nearest documents to synthetic examples and evaluates them using the evaluation LLM
3. **Active Learning Loop**:
   - Trains a classifier on labeled data
   - Predicts on all documents
   - Selects diverse uncertain samples for evaluation (stratified by confidence + diverse within each stratum)
   - LLM evaluates selected samples
   - Repeats until convergence or max iterations
4. **Early Stopping**: The process stops early if the flip rate (change in positive predictions) drops below 2%
5. **Results**: Filters and returns only positive items (prediction > 0.5) with scores

## Notes

- Embeddings are managed within the dataset - the same embedding model must be used for both dataset creation and classification
- The tool automatically validates embedding service consistency
- Each classification run is independent - you can run many queries on the same dataset
- Results are filtered to only include positive items (prediction > 0.5)
- The classifier uses stratified diverse sampling to ensure good coverage of the prediction space
