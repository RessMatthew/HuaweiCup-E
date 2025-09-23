import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TraditionalMLModels:
    """Traditional Machine Learning Models Manager"""

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}

    def initialize_models(self):
        """Initialize all traditional ML models with their configurations"""
        logger.info("Initializing traditional ML models")

        # Decision Tree
        self.models['decision_tree'] = DecisionTreeClassifier(**self.config.models['decision_tree'])

        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(**self.config.models['random_forest'])

        # AdaBoost
        self.models['adaboost'] = AdaBoostClassifier(**self.config.models['adaboost'])

        # Extra Trees
        self.models['extra_trees'] = ExtraTreesClassifier(**self.config.models['extra_trees'])

        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(**self.config.models['gradient_boosting'])

        # KNN
        self.models['knn'] = KNeighborsClassifier(**self.config.models['knn'])

        # SVM
        self.models['svm'] = SVC(**self.config.models['svm'])

        logger.info(f"Initialized {len(self.models)} traditional ML models")

    def initialize_external_models(self):
        """Initialize models that require external libraries"""
        try:
            # CatBoost
            from catboost import CatBoostClassifier
            self.models['catboost'] = CatBoostClassifier(**self.config.models['catboost'])
            logger.info("CatBoost model initialized")
        except ImportError:
            logger.warning("CatBoost not available. Install with: pip install catboost")

        try:
            # XGBoost
            from xgboost import XGBClassifier
            self.models['xgboost'] = XGBClassifier(**self.config.models['xgboost'])
            logger.info("XGBoost model initialized")
        except ImportError:
            logger.warning("XGBoost not available. Install with: pip install xgboost")

    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        logger.info(f"Training {model_name} model")
        model = self.models[model_name]

        try:
            model.fit(X_train, y_train)
            logger.info(f"{model_name} training completed")
            return model
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise

    def evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate a trained model"""
        y_pred = model.predict(X_val)

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1_score': f1_score(y_val, y_pred, average='weighted')
        }

        return metrics

    def cross_validate_model(self, model_name: str, X: np.ndarray, y: np.ndarray,
                           cv_splits: list) -> Dict[str, Any]:
        """Perform cross-validation for a specific model"""
        logger.info(f"Cross-validating {model_name} with {len(cv_splits)} folds")

        fold_results = []
        trained_models = []

        for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(cv_splits):
            logger.info(f"Training fold {fold_idx + 1}/{len(cv_splits)}")

            # Train model for this fold
            model = self.train_model(model_name, X_train, y_train)
            trained_models.append(model)

            # Evaluate on validation set
            metrics = self.evaluate_model(model, X_val, y_val)
            fold_results.append(metrics)

            logger.info(f"Fold {fold_idx + 1} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

        # Calculate average metrics
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            avg_metrics[metric] = np.mean([fold[metric] for fold in fold_results])
            avg_metrics[f'{metric}_std'] = np.std([fold[metric] for fold in fold_results])

        results = {
            'model_name': model_name,
            'avg_metrics': avg_metrics,
            'fold_results': fold_results,
            'trained_models': trained_models
        }

        logger.info(f"{model_name} - Average Accuracy: {avg_metrics['accuracy']:.4f} (±{avg_metrics['accuracy_std']:.4f})")
        logger.info(f"{model_name} - Average F1: {avg_metrics['f1_score']:.4f} (±{avg_metrics['f1_score_std']:.4f})")

        return results

    def get_model_info(self, model_name: str) -> str:
        """Get information about a specific model"""
        if model_name not in self.models:
            return f"Model {model_name} not available"

        model = self.models[model_name]
        return f"{model_name}: {type(model).__name__} with parameters: {model.get_params()}"

    def get_available_models(self) -> list:
        """Get list of available models"""
        return list(self.models.keys())