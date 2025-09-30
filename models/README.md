# Models Directory

This directory contains trained fraud detection models.

## Trained Models

After running `python src/train.py`, you'll find:

- `logistic_regression.pkl` - Logistic Regression model
- `random_forest.pkl` - Random Forest classifier
- `xgboost.pkl` - XGBoost classifier (usually best performing)
- `neural_network.h5` - Keras Neural Network
- `ensemble.pkl` - Ensemble voting classifier
- `preprocessor.pkl` - Data preprocessing pipeline (scalers and feature info)

## Model Sizes

Approximate file sizes:
- Logistic Regression: ~1 KB
- Random Forest: ~50-100 MB
- XGBoost: ~5-10 MB
- Neural Network: ~1-2 MB
- Ensemble: ~50-100 MB
- Preprocessor: ~1 KB

## Loading Models

### Python

```python
import joblib
from tensorflow import keras

# Load sklearn models
model = joblib.load('models/xgboost.pkl')

# Load Keras model
model = keras.models.load_model('models/neural_network.h5')

# Load preprocessor
preprocessor = joblib.load('models/preprocessor.pkl')
```

### Using the Predictor

```python
from src.predict import FraudPredictor

predictor = FraudPredictor(model_name='xgboost')
predictor.load_model()
result = predictor.predict_single(transaction)
```

## Model Performance

See `results/model_results.json` for detailed metrics.

Expected ROC-AUC scores:
- XGBoost: ~0.97
- Ensemble: ~0.97
- Random Forest: ~0.96
- Neural Network: ~0.96
- Logistic Regression: ~0.95

## Retraining

To retrain models with new data:

```bash
python src/train.py
```

This will overwrite existing models.