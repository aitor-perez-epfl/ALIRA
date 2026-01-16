"""
Active Learning Document Classifier package.

This package provides tools for building datasets with embeddings and
running active learning classification on documents.
"""

from .dataset_builder import DatasetBuilder
from .active_learner import ActiveLearner

__all__ = ['DatasetBuilder', 'ActiveLearner']
