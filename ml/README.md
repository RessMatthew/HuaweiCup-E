# Machine Learning Framework

A comprehensive machine learning framework that implements multiple models with K-fold cross-validation for classification tasks.

## Features

- **Multiple Models**: Decision Tree, Random Forest, AdaBoost, ExtraTrees, CatBoost, XGBoost, Gradient Boosting, and CNN
- **Cross-Validation**: Configurable K-fold cross-validation
- **macOS Support**: Optimized for macOS with MPS (Metal Performance Shaders) support
- **Easy Model Switching**: Command-line interface for model selection
- **Comprehensive Results**: Detailed metrics and summary reports

## Dataset

The framework expects a CSV file with the following structure:
- First column: `target_label` (classification labels)
- Other columns: Feature columns (23 features in total)
- Data should be pre-standardized

Default dataset path: `data/data_特征提取汇总_标准化.csv`

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Optional Dependencies

For full functionality, install additional packages:
```bash
pip install catboost xgboost
```

## Usage

### Command Line Options

```bash
./run_ml_framework.sh [OPTIONS]

Options:
  --model MODEL       Model to train (default: all)
                          Available models: all, decision_tree, random_forest,
                          adaboost, extra_trees, catboost, xgboost,
                          gradient_boosting, cnn

  --k-folds K         Number of folds for cross-validation (default: 5)

  --random-state R    Random state for reproducibility (default: 42)

  --device DEVICE     Device for PyTorch (default: mps for macOS)
                          Options: cpu, cuda, mps
```

### Direct Python Usage

You can also run the framework directly with Python:

```bash
# Run all models
python ml_framework.py

# Run specific model
python ml_framework.py --model random_forest --k_folds 5

# Run with custom parameters
python ml_framework.py --model cnn --k_folds 10 --device cpu --random_state 123
```

## Model Configurations

Models can be configured in `ml_config.py`:

### Traditional ML Models
- Decision Tree: max_depth, min_samples_split, etc.
- Random Forest: n_estimators, max_depth, etc.
- AdaBoost: n_estimators, learning_rate, etc.
- ExtraTrees: n_estimators, max_depth, etc.
- CatBoost: iterations, depth, learning_rate, etc.
- XGBoost: n_estimators, max_depth, learning_rate, etc.
- Gradient Boosting: n_estimators, max_depth, learning_rate, etc.

### CNN Model
- input_size: Number of features (23)
- hidden_size: Hidden layer size (128)
- dropout_rate: Dropout rate (0.3)
- learning_rate: Learning rate (0.001)
- batch_size: Batch size (32)
- epochs: Number of epochs (100)

## Output

### Results Directory
Results are saved in the `results/` directory:
- Individual model results: `{model_name}_results_{timestamp}.json`
- Summary report: `summary_report_{timestamp}.csv` and `summary_report_{timestamp}.json`

### Models Directory
Trained models are saved in the `models/` directory:
- Traditional ML models: `{model_name}/fold_{n}_{timestamp}.pkl`
- CNN models: `{model_name}/fold_{n}_{timestamp}.pth`

### Metrics
The framework reports the following metrics:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)

All metrics include standard deviation across folds.

## Architecture

### Core Components

1. **ml_config.py**: Configuration management
2. **data_utils.py**: Data loading and cross-validation utilities
3. **traditional_models.py**: Traditional ML models implementation
4. **cnn_model.py**: CNN neural network implementation
5. **ml_framework.py**: Unified training framework
6. **run_ml_framework.sh**: Shell script interface

### Model Architecture (CNN)

The CNN model consists of:
- 3 Convolutional layers with batch normalization and ReLU
- 3 Max pooling layers
- 2 Fully connected layers with dropout
- Cross-entropy loss function

## Troubleshooting

### Common Issues

1. **CatBoost/XGBoost not found**: Install with `pip install catboost xgboost`
2. **MPS device not available**: Use CPU with `-d cpu` flag
3. **Memory issues**: Reduce batch size in configuration
4. **Slow training**: Ensure MPS is available for macOS GPU acceleration

### Device Support

- **macOS**: Use `mps` for Metal Performance Shaders (GPU)
- **Linux/Windows with CUDA**: Use `cuda` for NVIDIA GPU
- **CPU fallback**: Use `cpu` for universal compatibility

## Requirements

- Python 3.7+
- PyTorch 1.9+
- scikit-learn 1.0+
- pandas 1.3+
- numpy 1.21+

## License

This framework is designed for educational and research purposes. Please ensure proper attribution when using this code."}

## File Structure

```
/Users/matthew/Workspace/HuaweiCup-E/
├── data/
│   └── data_特征提取汇总_标准化.csv          # Your dataset
├── ml_config.py                           # Configuration management
├── data_utils.py                          # Data loading utilities
├── traditional_models.py                  # Traditional ML models
├── cnn_model.py                           # CNN implementation
├── ml_framework.py                        # Main framework
├── run_ml_framework.sh                    # Shell script runner
├── requirements.txt                       # Python dependencies
├── results/                               # Results directory (created automatically)
└── models/                               # Models directory (created automatically)
```