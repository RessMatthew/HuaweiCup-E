import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from typing import Dict, Tuple, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNModel(nn.Module):
    """CNN Model for classification with 1D convolutions"""

    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128, dropout_rate: float = 0.3):
        super(CNNModel, self).__init__()

        # Input: (batch_size, 1, input_size)
        # Conv1D layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)  # -> (batch_size, 32, input_size)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # -> (batch_size, 32, input_size//2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)  # -> (batch_size, 64, input_size//2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # -> (batch_size, 64, input_size//4)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # -> (batch_size, 128, input_size//4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # -> (batch_size, 128, input_size//8)

        # Calculate the size after convolutions and pooling
        conv_output_size = (input_size // 8) * 128

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Add channel dimension for 1D convolution
        x = x.unsqueeze(1)  # (batch_size, input_size) -> (batch_size, 1, input_size)

        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # First fully connected layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        return x

class CNNTrainer:
    """Trainer for CNN model"""

    def __init__(self, config, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.device = torch.device(config.device if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def create_model(self, input_size: int) -> CNNModel:
        """Create CNN model"""
        model = CNNModel(
            input_size=input_size,
            num_classes=self.num_classes,
            hidden_size=self.config.models['cnn']['hidden_size'],
            dropout_rate=self.config.models['cnn']['dropout_rate']
        ).to(self.device)

        return model

    def train_epoch(self, model: CNNModel, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, model: CNNModel, val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float, Dict[str, float]]:
        """Validate the model"""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        return avg_loss, accuracy, metrics

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   input_size: int) -> Tuple[CNNModel, Dict[str, Any]]:
        """Train the CNN model"""
        logger.info("Training CNN model")

        # Create model
        model = self.create_model(input_size)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.models['cnn']['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        train_losses = []
        val_losses = []
        val_accuracies = []

        epochs = self.config.models['cnn']['epochs']
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)

            # Validate
            val_loss, val_acc, val_metrics = self.validate(model, val_loader, criterion)

            # Learning rate scheduling
            scheduler.step(val_loss)

            epoch_time = time.time() - start_time

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Log progress
            logger.info(f'Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s')
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            logger.info(f'Val Metrics - Precision: {val_metrics["precision"]:.4f}, '
                       f'Recall: {val_metrics["recall"]:.4f}, F1: {val_metrics["f1_score"]:.4f}')

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final validation
        final_val_loss, final_val_acc, final_metrics = self.validate(model, val_loader, criterion)

        training_results = {
            'best_val_acc': best_val_acc,
            'final_metrics': final_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'epochs_trained': len(train_losses)
        }

        logger.info(f"CNN training completed - Best validation accuracy: {best_val_acc:.4f}")

        return model, training_results

    def cross_validate_model(self, X: np.ndarray, y: np.ndarray, cv_splits: list) -> Dict[str, Any]:
        """Perform cross-validation for CNN model"""
        logger.info(f"Cross-validating CNN with {len(cv_splits)} folds")

        fold_results = []
        trained_models = []

        for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(cv_splits):
            logger.info(f"Training CNN fold {fold_idx + 1}/{len(cv_splits)}")

            # Create dataloaders
            from data_utils import CustomDataset
            train_dataset = CustomDataset(X_train, y_train)
            val_dataset = CustomDataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

            # Train model
            model, training_results = self.train_model(train_loader, val_loader, input_size=X_train.shape[1])

            trained_models.append(model)
            fold_results.append(training_results['final_metrics'])

            logger.info(f"Fold {fold_idx + 1} - Accuracy: {training_results['final_metrics']['accuracy']:.4f}, "
                       f"F1: {training_results['final_metrics']['f1_score']:.4f}")

        # Calculate average metrics
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            avg_metrics[metric] = np.mean([fold[metric] for fold in fold_results])
            avg_metrics[f'{metric}_std'] = np.std([fold[metric] for fold in fold_results])

        results = {
            'model_name': 'cnn',
            'avg_metrics': avg_metrics,
            'fold_results': fold_results,
            'trained_models': trained_models
        }

        logger.info(f"CNN - Average Accuracy: {avg_metrics['accuracy']:.4f} (±{avg_metrics['accuracy_std']:.4f})")
        logger.info(f"CNN - Average F1: {avg_metrics['f1_score']:.4f} (±{avg_metrics['f1_score_std']:.4f})")

        return results