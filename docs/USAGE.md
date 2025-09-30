# Usage Guide

## Quick Start

Train all models and generate visualizations with these simple commands:

```bash
# 1. Download/generate data (if not done during setup)
python src/download_data.py

# 2. Train all models
python src/train.py

# 3. Generate visualizations
python src/visualizations.py

# 4. Test predictions
python src/predict.py
```

## Detailed Usage

### 1. Data Preprocessing

The preprocessing pipeline is automatically run during training, but you can run it separately:

```bash
python src/data_preprocessing.py
```

This will:
- Load the credit card transaction data
- Engineer time-based and amount-based features
- Split data into train/validation/test sets (60/20/20)
- Scale features using StandardScaler and RobustScaler
- Save preprocessed data to `data/processed_data.pkl`

### 2. Model Training

Train all fraud detection models:

```bash
python src/train.py
```

This trains five models:
1. **Logistic Regression** - Fast, interpretable baseline
2. **Random Forest** - Ensemble decision trees
3. **XGBoost** - Gradient boosting (typically best performance)
4. **Neural Network** - Deep learning approach
5. **Ensemble** - Combines top 3 models

**Training Output:**
- Trained models saved to `models/`
- Results saved to `results/model_results.json`
- Shows accuracy, ROC-AUC, precision, recall, F1-score for each model

**Expected Training Time:**
- With synthetic data: 2-5 minutes
- With real dataset: 10-30 minutes (depending on hardware)

### 3. Generate Visualizations

Create comprehensive charts and plots:

```bash
python src/visualizations.py
```

This generates:
- Class distribution plots
- Confusion matrices for all models
- ROC curves comparison
- Precision-Recall curves
- Model performance comparison
- Feature importance (Random Forest)
- Summary dashboard

**Output:**
- Plots saved to `results/plots/`
- Also saved to `screenshots/` for GitHub display

### 4. Making Predictions

#### Interactive Demo

Run the prediction demo:

```bash
python src/predict.py
```

This demonstrates predictions on sample transactions.

#### Using in Your Code

```python
from src.predict import FraudPredictor

# Initialize predictor with your chosen model
predictor = FraudPredictor(model_name='xgboost')
predictor.load_model()

# Predict on a single transaction
transaction = {
    'Time': 12345,
    'V1': -1.359807,
    'V2': -0.072781,
    # ... (V3-V28)
    'Amount': 149.62
}

result = predictor.predict_single(transaction)
print(f"Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.4f}")
print(f"Risk Level: {result['risk_level']}")
```

#### Batch Predictions

```python
import pandas as pd
from src.predict import FraudPredictor

# Load transactions
df = pd.read_csv('new_transactions.csv')

# Initialize predictor
predictor = FraudPredictor(model_name='ensemble')
predictor.load_model()

# Make predictions
predictions, probabilities = predictor.predict(df)

# Add to DataFrame
df['is_fraud'] = predictions
df['fraud_probability'] = probabilities

# Save results
df.to_csv('predictions_output.csv', index=False)
```

## Model Selection

Choose a model based on your requirements:

| Model | Speed | Accuracy | Interpretability | Best For |
|-------|-------|----------|------------------|----------|
| Logistic Regression | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Fast, interpretable baseline |
| Random Forest | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Good balance, feature importance |
| XGBoost | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Best performance |
| Neural Network | ⚡ | ⭐⭐⭐⭐ | ⭐ | Complex patterns |
| Ensemble | ⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Maximum accuracy |

**Recommendation:**
- **Production use:** XGBoost or Ensemble
- **Real-time predictions:** Logistic Regression
- **Explainability needs:** Random Forest

## Advanced Configuration

### Custom Training Parameters

Edit model parameters in `src/models.py`:

```python
# Example: Increase Random Forest trees
rf_model = model_manager.create_random_forest(
    n_estimators=200,  # Default: 100
    class_weight='balanced'
)
```

### Handling Class Imbalance

The project uses multiple techniques:
- **Class weighting** (Logistic Regression, Random Forest)
- **scale_pos_weight** (XGBoost)
- **Class weights in loss** (Neural Network)

To modify:

```python
# In src/models.py
model = LogisticRegression(
    class_weight={0: 1, 1: 100}  # Custom weights
)
```

### Cross-Validation

Add cross-validation in `src/train.py`:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X_train, y_train,
    cv=5, scoring='f1'
)
print(f"CV F1-Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## Performance Metrics

The system reports multiple metrics:

- **Accuracy:** Overall correctness (less meaningful for imbalanced data)
- **ROC-AUC:** Area under ROC curve (0.5-1.0, higher is better)
- **PR-AUC:** Precision-Recall AUC (better for imbalanced data)
- **Precision (Fraud):** Of predicted frauds, how many are actually fraud
- **Recall (Fraud):** Of actual frauds, how many are detected
- **F1-Score (Fraud):** Harmonic mean of precision and recall

**Focus on PR-AUC and F1-Score** for fraud detection evaluation.

## Output Files

After running the complete pipeline:

```
ml-fraud-detection/
├── data/
│   ├── creditcard.csv           # Raw data
│   └── processed_data.pkl       # Preprocessed data
├── models/
│   ├── logistic_regression.pkl  # Trained models
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── neural_network.h5
│   ├── ensemble.pkl
│   └── preprocessor.pkl         # Scaler and feature info
├── results/
│   ├── model_results.json       # Performance metrics
│   └── plots/                   # All visualizations
└── screenshots/                 # For GitHub display
```

## Tips

1. **First run:** Use synthetic data to test the pipeline quickly
2. **Tuning:** Adjust hyperparameters based on validation metrics
3. **Production:** Retrain periodically with new fraud patterns
4. **Monitoring:** Track model performance over time
5. **Threshold:** Adjust prediction threshold based on precision/recall tradeoff

## Troubleshooting

**Training is slow:**
- Reduce dataset size for testing
- Use fewer estimators in ensemble models
- Train models individually

**Poor performance:**
- Check class distribution in your data
- Adjust class weights or use SMOTE
- Try different models

**Memory errors:**
- Use smaller batch sizes for Neural Network
- Reduce n_estimators for tree models
- Process data in chunks