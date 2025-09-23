import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse

# Import data utilities
from data_utils import DataLoaderManager
from ml_config import MLConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    """Model inference for loading and using trained models"""

    def __init__(self, config: MLConfig):
        self.config = config
        self.data_manager = DataLoaderManager(config)
        self.loaded_models = {}
        self.model_type = None

    def find_latest_models(self, model_name: str, num_folds: Optional[int] = None) -> List[str]:
        """Find the latest saved models for a specific model type"""
        models_dir = os.path.join(self.config.models_dir, model_name)

        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        # Get all model files
        model_files = []
        for file in os.listdir(models_dir):
            if file.endswith('.pkl') or file.endswith('.pth'):
                file_path = os.path.join(models_dir, file)
                try:
                    # Extract timestamp from filename (format: fold_X_YYYYMMDD_HHMMSS.ext)
                    parts = file.split('_')
                    if len(parts) >= 3:
                        date_part = parts[-2]  # YYYYMMDD
                        time_part = parts[-1].split('.')[0]  # HHMMSS
                        timestamp_str = f"{date_part}_{time_part}"
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        model_files.append((timestamp, file_path))
                except Exception as e:
                    logger.debug(f"Could not parse timestamp from {file}: {e}")
                    continue

        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")

        # Sort by timestamp (newest first)
        model_files.sort(key=lambda x: x[0], reverse=True)

        # Get the latest set of models (all folds from the same training run)
        if model_files:
            latest_timestamp = model_files[0][0]
            latest_models = [path for timestamp, path in model_files
                           if timestamp == latest_timestamp]

            if num_folds and len(latest_models) > num_folds:
                latest_models = latest_models[:num_folds]

            return latest_models

        return []

    def load_models(self, model_name: str, num_folds: Optional[int] = None):
        """Load trained models for inference"""
        logger.info(f"Loading models for {model_name}")

        model_files = self.find_latest_models(model_name, num_folds)
        self.loaded_models[model_name] = []
        self.model_type = model_name

        for model_file in model_files:
            try:
                if model_file.endswith('.pkl'):
                    # Load scikit-learn model
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    self.loaded_models[model_name].append(model)
                    logger.info(f"Loaded model from {model_file}")
                elif model_file.endswith('.pth'):
                    # Load PyTorch model (for CNN)
                    import torch
                    # This would need the CNN model architecture to be loaded properly
                    logger.info(f"PyTorch model found: {model_file} (CNN loading not implemented in this script)")
                else:
                    logger.warning(f"Unknown model file format: {model_file}")
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {str(e)}")
                raise

        if not self.loaded_models[model_name]:
            raise ValueError(f"No models were successfully loaded for {model_name}")

        logger.info(f"Successfully loaded {len(self.loaded_models[model_name])} models for {model_name}")

    def predict_single_model(self, model, X: np.ndarray) -> np.ndarray:
        """Make prediction with a single model"""
        try:
            if hasattr(model, 'predict'):
                return model.predict(X)
            else:
                raise ValueError("Model does not have predict method")
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict_ensemble(self, X: np.ndarray, voting: str = 'majority') -> np.ndarray:
        """Make predictions using ensemble of loaded models"""
        if not self.loaded_models or self.model_type not in self.loaded_models:
            raise ValueError("No models loaded. Please load models first.")

        models = self.loaded_models[self.model_type]
        logger.info(f"Making ensemble predictions with {len(models)} models")

        # Collect predictions from all models
        predictions = []
        for i, model in enumerate(models):
            try:
                pred = self.predict_single_model(model, X)
                predictions.append(pred)
                logger.info(f"Model {i+1} predictions completed")
            except Exception as e:
                logger.warning(f"Model {i+1} failed: {str(e)}")
                continue

        if not predictions:
            raise ValueError("No successful predictions from any model")

        # Combine predictions based on voting strategy
        predictions_array = np.array(predictions)

        if voting == 'majority':
            # Majority voting
            final_predictions = []
            for i in range(predictions_array.shape[1]):
                sample_predictions = predictions_array[:, i]
                unique, counts = np.unique(sample_predictions, return_counts=True)
                final_predictions.append(unique[np.argmax(counts)])
            return np.array(final_predictions)

        elif voting == 'average':
            # Average predictions (for regression or probability-based)
            return np.mean(predictions_array, axis=0)

        else:
            raise ValueError(f"Unknown voting strategy: {voting}")

    def predict_proba_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using ensemble of loaded models"""
        if not self.loaded_models or self.model_type not in self.loaded_models:
            raise ValueError("No models loaded. Please load models first.")

        models = self.loaded_models[self.model_type]
        logger.info(f"Making ensemble probability predictions with {len(models)} models")

        # Collect probability predictions from models that support it
        prob_predictions = []
        for i, model in enumerate(models):
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    prob_predictions.append(prob)
                    logger.info(f"Model {i+1} probability predictions completed")
            except Exception as e:
                logger.warning(f"Model {i+1} probability prediction failed: {str(e)}")
                continue

        if not prob_predictions:
            raise ValueError("No successful probability predictions from any model")

        # Average the probability predictions
        avg_probabilities = np.mean(prob_predictions, axis=0)
        return avg_probabilities

    def inference_from_csv(self, model_name: str, csv_path: str,
                          feature_columns: Optional[List[str]] = None,
                          voting: str = 'majority', evaluate: bool = True) -> pd.DataFrame:
        """Perform inference on data from CSV file"""
        logger.info(f"Loading data from {csv_path}")

        # Load data
        df = pd.read_csv(csv_path)

        # If feature columns not specified, use all except the first column (target)
        if feature_columns is None:
            # Assume the first column is the target variable
            feature_columns = df.columns[1:].tolist()  # Skip first column

        X = df[feature_columns].values
        logger.info(f"Using features: {feature_columns}")
        logger.info(f"Data shape: {X.shape}")

        # Load models
        self.load_models(model_name)

        # Make predictions
        predictions = self.predict_ensemble(X, voting=voting)

        # Create results dataframe
        results_df = df.copy()
        results_df['predicted_label'] = predictions

        # Add probability predictions if supported
        try:
            probabilities = self.predict_proba_ensemble(X)
            for i, class_name in enumerate([f'prob_class_{i}' for i in range(probabilities.shape[1])]):
                results_df[class_name] = probabilities[:, i]
        except Exception as e:
            logger.info(f"Probability predictions not available: {str(e)}")

        # Calculate accuracy if true labels are available
        if evaluate and len(df.columns) > 0:
            true_label_col = df.columns[0]  # Assume first column is true label
            if true_label_col in ['target_label', 'label', 'target', 'y'] or true_label_col.lower() in ['target_label', 'label', 'target', 'y']:
                try:
                    y_true = df[true_label_col].values
                    y_pred = predictions

                    # Calculate metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

                    # Store metrics in results
                    results_df['accuracy_score'] = accuracy
                    results_df['precision_score'] = precision
                    results_df['recall_score'] = recall
                    results_df['f1_score'] = f1

                    # Log metrics
                    logger.info("="*50)
                    logger.info("🎯 模型评估结果:")
                    logger.info(f"📊 准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
                    logger.info(f"📈 精确率 (Precision): {precision:.4f}")
                    logger.info(f"📉 召回率 (Recall): {recall:.4f}")
                    logger.info(f"🎯 F1分数 (F1-Score): {f1:.4f}")
                    logger.info("="*50)

                    # Add confusion matrix info
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_true, y_pred)
                    logger.info(f"混淆矩阵:")
                    logger.info(f"{cm}")

                except Exception as e:
                    logger.warning(f"无法计算评估指标: {e}")

        return results_df

    def inference_with_segment_voting(self, model_name: str, csv_path: str,
                                     feature_columns: Optional[List[str]] = None,
                                     voting: str = 'majority', segments_per_sample: int = 124) -> pd.DataFrame:
        """Perform inference with segment-based voting for unlabeled data"""
        logger.info(f"Loading segment data from {csv_path}")

        # Load data
        df = pd.read_csv(csv_path)
        total_rows = len(df)

        # Validate data structure
        if total_rows % segments_per_sample != 0:
            logger.warning(f"Total rows ({total_rows}) is not divisible by segments_per_sample ({segments_per_sample})")

        num_samples = total_rows // segments_per_sample
        logger.info(f"Found {num_samples} samples with {segments_per_sample} segments each")

        # If feature columns not specified, use all columns (no target column in unlabeled data)
        if feature_columns is None:
            feature_columns = df.columns.tolist()

        X = df[feature_columns].values
        logger.info(f"Using features: {feature_columns}")
        logger.info(f"Data shape: {X.shape}")

        # Load models
        self.load_models(model_name)

        # Make predictions for all segments
        segment_predictions = self.predict_ensemble(X, voting=voting)

        # Reshape predictions for voting
        segment_predictions = segment_predictions.reshape(num_samples, segments_per_sample)

        # Apply majority voting for each sample
        final_predictions = []
        segment_vote_counts = []  # Store vote counts for each sample

        for i in range(num_samples):
            sample_segments = segment_predictions[i, :]
            unique, counts = np.unique(sample_segments, return_counts=True)

            # Find the majority class
            majority_class = unique[np.argmax(counts)]
            final_predictions.append(majority_class)

            # Store vote counts as a dictionary
            vote_count = dict(zip(unique, counts))
            segment_vote_counts.append(vote_count)

            logger.info(f"Sample {i+1}: Majority class {majority_class}, Vote distribution: {vote_count}")

        # Create results dataframe with detailed voting information
        results_data = []
        for i in range(num_samples):
            vote_count = segment_vote_counts[i]
            for class_label, count in vote_count.items():
                results_data.append({
                    'sample_id': i + 1,
                    'predicted_label': final_predictions[i],
                    'total_segments': segments_per_sample,
                    'class_label': class_label,
                    'vote_count': count,
                    'vote_percentage': (count / segments_per_sample) * 100
                })

        results_df = pd.DataFrame(results_data)

        # Log summary
        logger.info("="*60)
        logger.info("📊 分段投票预测结果汇总:")
        final_pred_counts = pd.Series(final_predictions).value_counts()
        for label, count in final_pred_counts.items():
            percentage = (count / num_samples) * 100
            logger.info(f"类别 {label}: {count} 个样本 ({percentage:.1f}%)")
        logger.info("="*60)

        return results_df

def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='Model Inference Tool')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., xgboost, random_forest, knn)')
    parser.add_argument('--data', type=str, default='data/data_out.csv',
                       help='Path to CSV data file')
    parser.add_argument('--output', type=str, default='inference_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                       help='Feature column names (if not specified, will use all columns except target)')
    parser.add_argument('--voting', type=str, default='majority',
                       choices=['majority', 'average'],
                       help='Ensemble voting strategy')
    parser.add_argument('--num_folds', type=int, default=None,
                       help='Number of folds to use (default: use all available models)')

    args = parser.parse_args()

    # Create configuration
    config = MLConfig()

    # Initialize inference
    inference = ModelInference(config)

    try:
        # Perform inference
        logger.info(f"Starting inference with model: {args.model}")
        results = inference.inference_from_csv(
            model_name=args.model,
            csv_path=args.data,
            feature_columns=args.features,
            voting=args.voting
        )

        # Save results
        results.to_csv(args.output, index=False)
        logger.info(f"Inference completed. Results saved to: {args.output}")
        logger.info(f"Results shape: {results.shape}")

        # Print summary
        if 'predicted_label' in results.columns:
            pred_summary = results['predicted_label'].value_counts()
            print("\nPrediction Summary:")
            print(pred_summary)

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()