import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaggingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Bagging ensemble that integrates RandomForest, KNN, SVM, XGBoost, and CNN
    """

    def __init__(self, config, num_classes=None):
        self.config = config
        self.num_classes = num_classes
        self.models = {}
        self.ensemble = None
        self.is_fitted = False

    def _build_base_models(self):
        """Build the base models for the ensemble"""
        logger.info("Building base models for Bagging ensemble")

        # Random Forest
        self.models['random_forest'] = self.config.models['random_forest']

        # KNN
        self.models['knn'] = self.config.models['knn']

        # SVM
        self.models['svm'] = self.config.models['svm']

        # XGBoost
        try:
            from xgboost import XGBClassifier
            self.models['xgboost'] = self.config.models['xgboost']
        except ImportError:
            logger.warning("XGBoost not available for Bagging ensemble")
            self.models['xgboost'] = None

        # CNN will be handled separately due to its different nature
        logger.info(f"Built {len([m for m in self.models.values() if m is not None])} base models")

    def fit(self, X, y):
        """Fit the bagging ensemble"""
        logger.info("Fitting Bagging ensemble")

        self._build_base_models()

        # For simplicity, we'll use a voting approach with the fitted base models
        # In practice, you might want to implement a more sophisticated ensemble method

        self.fitted_models = {}

        # Fit Random Forest
        if self.models['random_forest'] is not None:
            rf_model = DecisionTreeClassifier(**self.config.models['decision_tree'])
            self.fitted_models['random_forest'] = rf_model.fit(X, y)

        # Fit KNN
        if self.models['knn'] is not None:
            from sklearn.neighbors import KNeighborsClassifier
            knn_model = KNeighborsClassifier(**self.config.models['knn'])
            self.fitted_models['knn'] = knn_model.fit(X, y)

        # Fit SVM
        if self.models['svm'] is not None:
            from sklearn.svm import SVC
            svm_model = SVC(**self.config.models['svm'])
            self.fitted_models['svm'] = svm_model.fit(X, y)

        # Fit XGBoost
        if self.models['xgboost'] is not None:
            try:
                from xgboost import XGBClassifier
                xgb_model = XGBClassifier(**self.config.models['xgboost'])
                self.fitted_models['xgboost'] = xgb_model.fit(X, y)
            except ImportError:
                logger.warning("XGBoost fitting skipped")

        self.is_fitted = True
        logger.info("Bagging ensemble fitting completed")
        return self

    def predict(self, X):
        """Make predictions using the ensemble"""
        if not self.is_fitted:
            raise ValueError("Bagging ensemble must be fitted before making predictions")

        logger.info("Making predictions with Bagging ensemble")

        # Collect predictions from all base models
        predictions = []

        for model_name, model in self.fitted_models.items():
            if model is not None:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {str(e)}")

        if not predictions:
            raise ValueError("No successful predictions from base models")

        # Use majority voting
        predictions_array = np.array(predictions)
        majority_vote = []

        for i in range(predictions_array.shape[1]):
            sample_predictions = predictions_array[:, i]
            # Get the most common prediction
            unique, counts = np.unique(sample_predictions, return_counts=True)
            majority_vote.append(unique[np.argmax(counts)])

        return np.array(majority_vote)

    def predict_proba(self, X):
        """Predict class probabilities (if supported by base models)"""
        if not self.is_fitted:
            raise ValueError("Bagging ensemble must be fitted before making predictions")

        logger.info("Making probability predictions with Bagging ensemble")

        # Collect probability predictions from models that support it
        prob_predictions = []

        for model_name, model in self.fitted_models.items():
            if model is not None and hasattr(model, 'predict_proba'):
                try:
                    prob = model.predict_proba(X)
                    prob_predictions.append(prob)
                except Exception as e:
                    logger.warning(f"Probability prediction failed for {model_name}: {str(e)}")

        if not prob_predictions:
            # Fallback to hard voting probabilities
            hard_predictions = self.predict(X)
            n_classes = len(np.unique(hard_predictions))
            prob_predictions = np.zeros((len(hard_predictions), n_classes))
            for i, pred in enumerate(hard_predictions):
                prob_predictions[i, pred] = 1.0
            return prob_predictions

        # Average the probability predictions
        avg_probabilities = np.mean(prob_predictions, axis=0)
        return avg_probabilities


class BaggingEnsembleTrainer:
    """Trainer for Bagging ensemble model"""

    def __init__(self, config):
        self.config = config
        self.ensemble_model = None

    def cross_validate_model(self, X, y, cv_splits):
        """Perform cross-validation for the bagging ensemble"""
        logger.info("Cross-validating Bagging ensemble")

        fold_results = []
        trained_models = []

        for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(cv_splits):
            logger.info(f"Training ensemble fold {fold_idx + 1}/{len(cv_splits)}")

            # Create and train ensemble for this fold
            num_classes = len(np.unique(y))
            ensemble = BaggingEnsemble(self.config, num_classes=num_classes)

            try:
                # Train ensemble
                ensemble.fit(X_train, y_train)
                trained_models.append(ensemble)

                # Evaluate on validation set
                y_pred = ensemble.predict(X_val)

                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, average='weighted'),
                    'recall': recall_score(y_val, y_pred, average='weighted'),
                    'f1_score': f1_score(y_val, y_pred, average='weighted')
                }

                fold_results.append(metrics)
                logger.info(f"Fold {fold_idx + 1} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

            except Exception as e:
                logger.error(f"Error in fold {fold_idx + 1}: {str(e)}")
                fold_results.append({
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                    'error': str(e)
                })

        # Calculate average metrics
        avg_metrics = {}
        valid_folds = [fold for fold in fold_results if 'error' not in fold]

        if valid_folds:
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                avg_metrics[metric] = np.mean([fold[metric] for fold in valid_folds])
                avg_metrics[f'{metric}_std'] = np.std([fold[metric] for fold in valid_folds])
        else:
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                avg_metrics[metric] = 0.0
                avg_metrics[f'{metric}_std'] = 0.0

        results = {
            'model_name': 'bagging_ensemble',
            'avg_metrics': avg_metrics,
            'fold_results': fold_results,
            'trained_models': trained_models
        }

        logger.info(f"Bagging ensemble - Average Accuracy: {avg_metrics['accuracy']:.4f} (±{avg_metrics['accuracy_std']:.4f})")
        logger.info(f"Bagging ensemble - Average F1: {avg_metrics['f1_score']:.4f} (±{avg_metrics['f1_score_std']:.4f})")

        return results