# Results Directory

This directory contains model evaluation results and visualizations.

## Contents

### Metrics
- `model_results.json` - Detailed performance metrics for all models

### Plots (`results/plots/`)
- `class_distribution.png` - Distribution of fraud vs legitimate transactions
- `confusion_matrices.png` - Confusion matrices for all models
- `roc_curves.png` - ROC curves comparison
- `precision_recall_curves.png` - Precision-Recall curves comparison
- `model_comparison.png` - Side-by-side metric comparison
- `feature_importance.png` - Top features for Random Forest
- `summary_dashboard.png` - Complete performance dashboard

## Generate Results

Run the visualization script after training:

```bash
python src/visualizations.py
```

## Model Results JSON Structure

```json
{
  "xgboost": {
    "test_accuracy": 0.9995,
    "test_roc_auc": 0.9742,
    "test_pr_auc": 0.8512,
    "test_precision_fraud": 0.9084,
    "test_recall_fraud": 0.8333,
    "test_f1_fraud": 0.8693,
    "confusion_matrix": [[56862, 2], [8, 40]]
  },
  ...
}
```

## Interpreting Results

### Key Metrics

- **ROC-AUC**: Overall discrimination ability (higher is better, 0.5-1.0)
- **PR-AUC**: Precision-Recall AUC (better for imbalanced data)
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Of predicted frauds, how many are actually fraud
- **Recall**: Of actual frauds, how many are detected

### Focus Areas

For fraud detection, prioritize:
1. **PR-AUC** - More meaningful than ROC-AUC for imbalanced data
2. **Recall** - Minimize missed frauds (false negatives)
3. **Precision** - Minimize false alarms (false positives)

The optimal model balances precision and recall based on business requirements.