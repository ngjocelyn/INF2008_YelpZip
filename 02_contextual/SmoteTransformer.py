from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin

class SMOTETransformer():
    def __init__(self, sampling_strategy='auto', random_state=42):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state)

    def fit_resample(self, X, y=None):
        """Apply SMOTE only when both X and y are available (training data)."""
        if y is not None:
            X_resampled = self.smote.fit_resample(X, y)
            return X_resampled
        return X
