# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ free disk space (for dataset and models)
- (Optional) Kaggle API credentials for real dataset

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ml-fraud-detection.git
cd ml-fraud-detection
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy (data processing)
- scikit-learn (ML algorithms)
- xgboost (gradient boosting)
- tensorflow (neural networks)
- matplotlib, seaborn, plotly (visualization)
- imbalanced-learn (handling imbalanced data)
- kaggle (optional, for dataset download)

### 4. Download Dataset

#### Option A: Using Kaggle API (Recommended for Real Data)

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account settings → API → Create New API Token
3. This downloads `kaggle.json` - place it in `~/.kaggle/`
4. Run the download script:

```bash
python src/download_data.py
```

#### Option B: Synthetic Data (For Testing)

If you don't have Kaggle credentials, the script will automatically generate synthetic data:

```bash
python src/download_data.py
```

The script will create a synthetic dataset that mimics the structure of the real Credit Card Fraud Detection dataset.

### 5. Verify Installation

Test that everything is installed correctly:

```bash
# Test model creation
python src/models.py

# Test preprocessing
python -c "from src.data_preprocessing import FraudDataPreprocessor; print('✓ Setup successful!')"
```

## Troubleshooting

### ModuleNotFoundError

If you get import errors, make sure your virtual environment is activated and all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Kaggle API Issues

If Kaggle download fails:
- Verify `kaggle.json` is in `~/.kaggle/` directory
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
- The script will automatically fall back to synthetic data

### Memory Issues

If training fails due to memory:
- Use a smaller dataset by modifying `src/download_data.py`
- Reduce `n_estimators` in model configurations
- Train models individually instead of all at once

### TensorFlow Warnings

You may see TensorFlow warnings about CPU optimization - these can be safely ignored for this project.

## Next Steps

After successful installation:
1. Train models: See [USAGE.md](USAGE.md)
2. View examples: See [EXAMPLES.md](EXAMPLES.md)