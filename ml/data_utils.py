import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    """PyTorch Dataset for neural network models"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataLoaderManager:
    """Manages data loading, preprocessing, and cross-validation splits"""

    def __init__(self, config):
        self.config = config
        self.data = None
        self.X = None
        self.y = None
        self.label_encoder = LabelEncoder()

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file"""
        logger.info(f"Loading data from {self.config.data_path}")

        try:
            self.data = pd.read_csv(self.config.data_path)
            logger.info(f"Data shape: {self.data.shape}")

            # Separate features and target
            self.X = self.data.drop('target_label', axis=1).values
            self.y = self.data['target_label'].values

            # Encode labels if they are not numeric
            if not np.issubdtype(self.y.dtype, np.number):
                self.y = self.label_encoder.fit_transform(self.y)
                logger.info(f"Label encoding mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

            logger.info(f"Features shape: {self.X.shape}, Target shape: {self.y.shape}")
            logger.info(f"Number of classes: {len(np.unique(self.y))}")
            logger.info(f"Class distribution: {np.bincount(self.y)}")

            return self.X, self.y

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def get_kfold_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Generate K-fold cross-validation splits"""
        skf = StratifiedKFold(n_splits=self.config.k_folds, shuffle=True, random_state=self.config.random_state)

        splits = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            splits.append((X_train, X_val, y_train, y_val))

        logger.info(f"Generated {len(splits)} cross-validation splits")
        return splits

    def get_pytorch_dataloaders(self, X_train: np.ndarray, X_val: np.ndarray,
                               y_train: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for training and validation"""

        # Create datasets
        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for macOS compatibility
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0  # Set to 0 for macOS compatibility
        )

        return train_loader, val_loader

    def get_feature_names(self) -> List[str]:
        """Get feature names from the dataset"""
        if self.data is not None:
            return [col for col in self.data.columns if col != 'target_label']
        return []

    def get_class_names(self) -> List[str]:
        """Get original class names if labels were encoded"""
        if hasattr(self.label_encoder, 'classes_'):
            return list(self.label_encoder.classes_)
        return [str(i) for i in range(len(np.unique(self.y)))]