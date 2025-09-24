import os
import json
import pickle
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, List
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from ml_config import MLConfig
from data_utils import DataLoaderManager
from traditional_models import TraditionalMLModels
from cnn_model import CNNTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLFramework:
    """Unified Machine Learning Framework for multiple models"""

    def __init__(self, config: MLConfig):
        self.config = config
        self.data_manager = DataLoaderManager(config)
        self.traditional_ml = TraditionalMLModels(config)
        self.cnn_trainer = None
        self.results = {}

    def setup_models(self):
        """Initialize all available models"""
        logger.info("Setting up models")
        self.traditional_ml.initialize_models()
        self.traditional_ml.initialize_external_models()

    def run_single_model(self, model_name: str, X: np.ndarray, y: np.ndarray,
                        cv_splits: List) -> Dict[str, Any]:
        """Run cross-validation for a single model"""
        logger.info(f"Running model: {model_name}")

        try:
            if model_name == 'cnn':
                # Handle CNN separately
                if self.cnn_trainer is None:
                    self.cnn_trainer = CNNTrainer(self.config, num_classes=len(np.unique(y)))
                results = self.cnn_trainer.cross_validate_model(X, y, cv_splits)
            else:
                # Handle traditional ML models
                results = self.traditional_ml.cross_validate_model(model_name, X, y, cv_splits)

            # Save trained models if available
            if 'trained_models' in results and results['trained_models']:
                self.save_trained_models(model_name, results['trained_models'])

            return results

        except Exception as e:
            logger.error(f"Error running {model_name}: {str(e)}")
            return {'model_name': model_name, 'error': str(e)}

    def run_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run cross-validation for all available models"""
        logger.info("Running all models")

        # Generate cross-validation splits
        cv_splits = self.data_manager.get_kfold_splits(X, y)

        # Convert to the format expected by traditional ML models
        cv_splits_formatted = []
        for X_train, X_val, y_train, y_val in cv_splits:
            cv_splits_formatted.append((X_train, X_val, y_train, y_val))

        all_results = {}

        # Get available models
        available_models = self.traditional_ml.get_available_models()
        available_models.append('cnn')  # Add CNN

        for model_name in available_models:
            try:
                logger.info(f"Processing model: {model_name}")
                results = self.run_single_model(model_name, X, y, cv_splits_formatted)
                all_results[model_name] = results

                # Save individual model results
                self.save_model_results(model_name, results)

            except Exception as e:
                logger.error(f"Failed to process {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}

        self.results = all_results
        return all_results

    def save_model_results(self, model_name: str, results: Dict[str, Any]):
        """Save results for a specific model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.config.results_dir, f"{model_name}_results_{timestamp}.json")

        # Prepare results for JSON serialization
        json_results = self.prepare_results_for_json(results)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved results for {model_name} to {results_file}")

    def prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results to JSON-serializable format"""
        json_results = {}

        for key, value in results.items():
            if key == 'trained_models':
                # Skip saving trained models in JSON (they're saved separately)
                json_results[key] = f"{len(value)} models trained"
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = self.prepare_results_for_json(value)
            elif isinstance(value, (np.integer, np.floating)):
                json_results[key] = value.item()
            else:
                json_results[key] = value

        return json_results

    def save_trained_models(self, model_name: str, trained_models: List):
        """Save trained models to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = os.path.join(self.config.models_dir, model_name)
        os.makedirs(models_dir, exist_ok=True)

        for idx, model in enumerate(trained_models):
            if model_name == 'cnn':
                # Save PyTorch model
                model_path = os.path.join(models_dir, f"fold_{idx+1}_{timestamp}.pth")
                torch.save(model.state_dict(), model_path)
            else:
                # Save scikit-learn model
                model_path = os.path.join(models_dir, f"fold_{idx+1}_{timestamp}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

        logger.info(f"Saved {len(trained_models)} trained {model_name} models")

    def generate_summary_report(self) -> pd.DataFrame:
        """Generate summary report of all model results"""
        summary_data = []

        for model_name, results in self.results.items():
            if 'avg_metrics' in results:
                avg_metrics = results['avg_metrics']
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': f"{avg_metrics['accuracy']:.4f} (±{avg_metrics.get('accuracy_std', 0):.4f})",
                    'Precision': f"{avg_metrics['precision']:.4f} (±{avg_metrics.get('precision_std', 0):.4f})",
                    'Recall': f"{avg_metrics['recall']:.4f} (±{avg_metrics.get('recall_std', 0):.4f})",
                    'F1-Score': f"{avg_metrics['f1_score']:.4f} (±{avg_metrics.get('f1_score_std', 0):.4f})"
                })
            elif 'error' in results:
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': 'ERROR',
                    'Precision': results['error'],
                    'Recall': '',
                    'F1-Score': ''
                })

        return pd.DataFrame(summary_data)

    def save_summary_report(self):
        """Save summary report to CSV and JSON"""
        if not self.results:
            logger.warning("No results to save")
            return

        summary_df = self.generate_summary_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as CSV
        csv_file = os.path.join(self.config.results_dir, f"summary_report_{timestamp}.csv")
        summary_df.to_csv(csv_file, index=False)
        logger.info(f"Saved summary report to {csv_file}")

        # Save as JSON
        json_file = os.path.join(self.config.results_dir, f"summary_report_{timestamp}.json")
        summary_df.to_json(json_file, orient='records', indent=2)
        logger.info(f"Saved summary report to {json_file}")

        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)

    def plot_roc_curves(self, save_plots: bool = True):
        """Generate ROC curves for all models with AUC values"""
        logger.info("Generating ROC curves and AUC values")

        # Color scheme from .claude/配色方案.md
        colors = ['#237B9F', '#71BFB2', '#AD0B08', '#EC817E', '#FEE066']
        color_alpha = 0.85

        plt.figure(figsize=(7, 7))  # width = 7 inches as specified

        color_idx = 0
        auc_values = {}

        for model_name, results in self.results.items():
            if 'fold_results' not in results or 'error' in results:
                continue

            try:
                # Collect predictions and true labels from all folds
                all_y_true = []
                all_y_scores = []

                for fold_result in results['fold_results']:
                    if 'y_true' in fold_result and 'y_scores' in fold_result:
                        all_y_true.extend(fold_result['y_true'])
                        all_y_scores.extend(fold_result['y_scores'])

                if not all_y_true or not all_y_scores:
                    logger.warning(f"No prediction data available for {model_name}")
                    continue

                # Convert to numpy arrays
                y_true = np.array(all_y_true)
                y_scores = np.array(all_y_scores)

                # Handle binary classification
                if len(np.unique(y_true)) == 2:
                    # For binary classification
                    if y_scores.ndim > 1 and y_scores.shape[1] == 2:
                        y_scores = y_scores[:, 1]  # Use probabilities for positive class

                    fpr, tpr, _ = roc_curve(y_true, y_scores)
                    roc_auc = auc(fpr, tpr)

                else:
                    # For multi-class classification, use micro-average
                    n_classes = len(np.unique(y_true))
                    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

                    if y_scores.ndim == 1:
                        logger.warning(f"Single dimension scores for multi-class {model_name}, skipping")
                        continue

                    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
                    roc_auc = auc(fpr, tpr)

                # Store AUC value
                auc_values[model_name] = roc_auc

                # Plot ROC curve
                color = colors[color_idx % len(colors)]
                plt.plot(fpr, tpr, color=color, alpha=color_alpha,
                        lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

                color_idx += 1

            except Exception as e:
                logger.error(f"Error generating ROC curve for {model_name}: {str(e)}")
                continue

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)

        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Save plot
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(self.config.results_dir, f"roc_curves_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved ROC curves plot to {plot_file}")

        # Print AUC values summary
        print("\n" + "="*50)
        print("AUC VALUES SUMMARY")
        print("="*50)
        for model_name, auc_val in sorted(auc_values.items(), key=lambda x: x[1], reverse=True):
            print(f"{model_name:<20}: {auc_val:.4f}")
        print("="*50)

        return auc_values

def main():
    """Main function to run the ML framework"""
    parser = argparse.ArgumentParser(description='Machine Learning Framework')
    parser.add_argument('--model', type=str, choices=['all', 'decision_tree', 'random_forest', 'adaboost',
                                                     'extra_trees', 'catboost', 'xgboost', 'gradient_boosting', 'knn', 'svm', 'bagging_ensemble', 'cnn'],
                       default='all', help='Model to train (default: all)')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility (default: 42)')
    parser.add_argument('--device', type=str, default='mps', help='Device for PyTorch (default: mps for macOS)')
    parser.add_argument('--input_size', type=int, help='Input size for CNN model (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs for CNN model (overrides config)')

    args = parser.parse_args()

    # Create configuration
    config = MLConfig(
        k_folds=args.k_folds,
        random_state=args.random_state,
        device=args.device
    )

    # Override CNN parameters if provided
    if args.input_size is not None:
        config.models['cnn']['input_size'] = args.input_size
    if args.epochs is not None:
        config.models['cnn']['epochs'] = args.epochs

    # Initialize framework
    framework = MLFramework(config)
    framework.setup_models()

    # Load data
    logger.info("Loading data...")
    X, y = framework.data_manager.load_data()

    # Run models
    if args.model == 'all':
        logger.info("Running all models...")
        results = framework.run_all_models(X, y)
    else:
        logger.info(f"Running {args.model}...")
        cv_splits = framework.data_manager.get_kfold_splits(X, y)
        cv_splits_formatted = [(X_train, X_val, y_train, y_val) for X_train, X_val, y_train, y_val in cv_splits]
        results = framework.run_single_model(args.model, X, y, cv_splits_formatted)
        framework.results = {args.model: results}

    # Save summary report
    framework.save_summary_report()

    # Generate ROC curves and AUC values
    framework.plot_roc_curves()

    logger.info("Training completed!")

if __name__ == "__main__":
    import torch
    main()