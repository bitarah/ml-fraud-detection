# Examples

## Example 1: Basic Training and Prediction

Complete workflow from data to predictions:

```bash
# Step 1: Setup data
python src/download_data.py

# Step 2: Train models
python src/train.py

# Step 3: Make predictions
python src/predict.py
```

**Expected Output:**
```
✓ Dataset downloaded/generated
✓ 5 models trained with metrics
✓ Sample predictions with risk levels
```

## Example 2: Train a Single Model

Train only XGBoost (fastest and most accurate):

```python
from src.data_preprocessing import FraudDataPreprocessor
from src.models import FraudDetectionModels
import joblib

# Load or create preprocessed data
preprocessor = FraudDataPreprocessor()
data_splits = preprocessor.prepare_data()

# Train only XGBoost
model_manager = FraudDetectionModels()
xgb_model = model_manager.create_xgboost(scale_pos_weight=100)

model, metrics = model_manager.train_model(
    xgb_model,
    data_splits['X_train'],
    data_splits['y_train'],
    data_splits['X_val'],
    data_splits['y_val'],
    model_name='XGBoost'
)

# Evaluate on test set
test_metrics = model_manager.evaluate_model(
    xgb_model,
    data_splits['X_test'],
    data_splits['y_test'],
    'XGBoost'
)

print(f"Test F1-Score: {test_metrics['test_f1_fraud']:.4f}")
```

## Example 3: Custom Prediction Pipeline

Build a custom prediction workflow:

```python
import pandas as pd
from src.predict import FraudPredictor

# Load your transaction data
transactions = pd.read_csv('transactions.csv')

# Initialize predictor with best model
predictor = FraudPredictor(model_name='xgboost')
predictor.load_model()

# Make predictions
predictions, probabilities = predictor.predict(transactions)

# Filter high-risk transactions
high_risk = transactions[probabilities > 0.7].copy()
high_risk['fraud_probability'] = probabilities[probabilities > 0.7]

# Save for manual review
high_risk.to_csv('high_risk_transactions.csv', index=False)

print(f"Found {len(high_risk)} high-risk transactions")
print(f"Fraud rate: {(probabilities > 0.5).mean() * 100:.2f}%")
```

## Example 4: Feature Importance Analysis

Understand which features drive fraud detection:

```python
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load trained Random Forest model
rf_model = joblib.load('models/random_forest.pkl')

# Load feature names
data = joblib.load('data/processed_data.pkl')
features = data['feature_columns']

# Get feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

# Plot top 10 features
plt.figure(figsize=(10, 6))
plt.bar(range(10), importances[indices])
plt.xticks(range(10), [features[i] for i in indices], rotation=45)
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

## Example 5: Real-Time Fraud Scoring API

Create a simple API for fraud detection:

```python
from flask import Flask, request, jsonify
from src.predict import FraudPredictor

app = Flask(__name__)

# Load model once at startup
predictor = FraudPredictor(model_name='xgboost')
predictor.load_model()

@app.route('/predict', methods=['POST'])
def predict_fraud():
    # Get transaction data from request
    transaction = request.json

    # Make prediction
    result = predictor.predict_single(transaction)

    # Return result
    return jsonify({
        'transaction_id': transaction.get('id'),
        'is_fraud': result['is_fraud'],
        'fraud_probability': result['fraud_probability'],
        'risk_level': result['risk_level'],
        'confidence': result['confidence']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Usage:**
```bash
# Start API
python fraud_api.py

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "id": "12345",
    "Time": 12345,
    "V1": -1.359807,
    "Amount": 149.62,
    ...
  }'
```

## Example 6: Model Comparison Notebook

Compare all models interactively:

```python
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load results
with open('results/model_results.json', 'r') as f:
    results = json.load(f)

# Create comparison DataFrame
comparison = pd.DataFrame([
    {
        'Model': name.replace('_', ' ').title(),
        'ROC-AUC': metrics['test_roc_auc'],
        'PR-AUC': metrics['test_pr_auc'],
        'F1-Score': metrics['test_f1_fraud'],
        'Precision': metrics['test_precision_fraud'],
        'Recall': metrics['test_recall_fraud']
    }
    for name, metrics in results.items()
])

# Display
print(comparison.to_string(index=False))

# Plot comparison
comparison.set_index('Model')[['ROC-AUC', 'PR-AUC', 'F1-Score']].plot(
    kind='bar', figsize=(12, 6), rot=45
)
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

## Example 7: Hyperparameter Tuning

Optimize model parameters:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load preprocessed data
data = joblib.load('data/processed_data.pkl')
X_train = data['X_train']
y_train = data['y_train']

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10, 20]
}

# Create model
rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Grid search with cross-validation
grid_search = GridSearchCV(
    rf, param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Fit
grid_search.fit(X_train, y_train)

# Results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1-score: {grid_search.best_score_:.4f}")

# Save best model
joblib.dump(grid_search.best_estimator_, 'models/rf_tuned.pkl')
```

## Example 8: Threshold Optimization

Find optimal classification threshold:

```python
import numpy as np
import joblib
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load('models/xgboost.pkl')
data = joblib.load('data/processed_data.pkl')
X_val = data['X_val']
y_val = data['y_val']

# Get probabilities
y_proba = model.predict_proba(X_val)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

# Calculate F1 for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

# Find optimal threshold
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"F1-score at optimal: {f1_scores[optimal_idx]:.4f}")
print(f"Precision: {precision[optimal_idx]:.4f}")
print(f"Recall: {recall[optimal_idx]:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.plot(thresholds, f1_scores[:-1], label='F1-Score', linewidth=2)
plt.axvline(optimal_threshold, color='r', linestyle='--', label='Optimal')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Optimization')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## Example 9: Batch Processing

Process large datasets in batches:

```python
import pandas as pd
from src.predict import FraudPredictor
from tqdm import tqdm

# Initialize predictor
predictor = FraudPredictor(model_name='xgboost')
predictor.load_model()

# Process in batches
batch_size = 10000
results = []

for chunk in tqdm(pd.read_csv('large_dataset.csv', chunksize=batch_size)):
    predictions, probabilities = predictor.predict(chunk)

    chunk['is_fraud'] = predictions
    chunk['fraud_probability'] = probabilities

    # Save high-risk only (to save space)
    high_risk = chunk[probabilities > 0.5]
    results.append(high_risk)

# Combine results
all_high_risk = pd.concat(results, ignore_index=True)
all_high_risk.to_csv('batch_predictions_high_risk.csv', index=False)

print(f"Processed {len(results) * batch_size:,} transactions")
print(f"Found {len(all_high_risk):,} high-risk transactions")
```

## Example 10: Model Monitoring

Track model performance over time:

```python
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, f1_score
from datetime import datetime

# Load model
model = joblib.load('models/xgboost.pkl')

# Load new data (with ground truth labels)
new_data = pd.read_csv('new_transactions_with_labels.csv')

# Separate features and labels
X_new = new_data.drop(['Class'], axis=1)
y_new = new_data['Class']

# Make predictions
y_pred = model.predict(X_new)
y_proba = model.predict_proba(X_new)[:, 1]

# Calculate metrics
metrics = {
    'date': datetime.now().strftime('%Y-%m-%d'),
    'roc_auc': roc_auc_score(y_new, y_proba),
    'f1_score': f1_score(y_new, y_pred),
    'fraud_rate': y_new.mean(),
    'n_transactions': len(y_new)
}

# Log to file
log_df = pd.DataFrame([metrics])
log_df.to_csv('model_monitoring.csv', mode='a', header=False, index=False)

print(f"Model Performance on {metrics['date']}:")
print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"  F1-Score: {metrics['f1_score']:.4f}")

# Alert if performance degrades
if metrics['roc_auc'] < 0.90:
    print("⚠️  WARNING: Model performance degraded. Consider retraining.")
```

## Demo Videos/Images

See the `screenshots/` directory for:
- Model comparison dashboard
- ROC curves for all models
- Confusion matrices
- Feature importance plots
- Precision-recall curves