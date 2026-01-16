from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression


class AbstractClassifier(ABC):
    """Abstract base class for classifiers."""
    
    @abstractmethod
    def fit(self, X, y):
        """Train the classifier."""
        raise NotImplementedError
        
    @abstractmethod
    def predict_proba(self, X):
        """Predict class probabilities."""
        raise NotImplementedError


class LogisticRegressionClassifier(AbstractClassifier):
    """Logistic regression classifier implementation."""
    
    def __init__(self, c=1.0):
        self.c = c
        self.model = LogisticRegression(
            C=c,
            penalty="l2",
            max_iter=10000,
            solver="saga",
            class_weight="balanced"
        )
    
    def fit(self, X, y):
        """Train the logistic regression model."""
        self.model.fit(X, y)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
