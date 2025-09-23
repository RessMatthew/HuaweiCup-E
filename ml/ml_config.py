import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MLConfig:
    # Data paths
    data_path: str = "data/data_out.csv"

    # Cross-validation settings
    k_folds: int = 5
    random_state: int = 42

    # Model configurations
    models: Dict[str, Dict[str, Any]] = None

    # Training settings
    device: str = "mps"  # For macOS GPU acceleration
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001

    # Results settings
    results_dir: str = "results"
    models_dir: str = "models"

    def __post_init__(self):
        if self.models is None:
            self.models = {
                'decision_tree': {
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': self.random_state
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': self.random_state,
                    'n_jobs': -1
                },
                'adaboost': {
                    'n_estimators': 100,
                    'learning_rate': 1.0,
                    'random_state': self.random_state
                },
                'extra_trees': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': self.random_state,
                    'n_jobs': -1
                },
                'catboost': {
                    'iterations': 100,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'random_seed': self.random_state,
                    'verbose': False
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': self.random_state,
                    'n_jobs': -1
                },
                'gradient_boosting': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': self.random_state
                },
                'knn': {
                    'n_neighbors': 5,
                    'weights': 'uniform',
                    'metric': 'minkowski',
                    'p': 2  # p=2 for Euclidean distance
                },
                'svm': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'random_state': self.random_state
                },
                'bagging_ensemble': {
                    'n_estimators': 5,  # Number of base models
                    'max_samples': 1.0,  # Fraction of samples to draw
                    'max_features': 1.0,  # Fraction of features to draw
                    'bootstrap': True,
                    'bootstrap_features': False,
                    'oob_score': False,
                    'random_state': self.random_state,
                    'n_jobs': -1,
                    'verbose': 0
                },
                'cnn': {
                    'input_size': 23,  # Number of features
                    'hidden_size': 128,
                    'num_classes': None,  # Will be determined from data
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'epochs': 20
                }
            }

        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)