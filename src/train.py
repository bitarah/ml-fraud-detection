"""
Training Script for Fraud Detection Models
Trains multiple models and compares their performance
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import FraudDataPreprocessor
from models import FraudDetectionModels

def train_all_models(data_splits, random_state=42):
    """
    Train all fraud detection models

    Args:
        data_splits: Dictionary containing train/val/test splits
        random_state: Random seed

    Returns:
        Dictionary of trained models and results
    """
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']

    # Calculate scale_pos_weight for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    print(f"Scale pos weight (for XGBoost): {scale_pos_weight:.2f}")

    # Initialize model manager
    model_manager = FraudDetectionModels(random_state=random_state)

    # Dictionary to store all results
    all_results = {}

    # ========== Train Logistic Regression ==========
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    lr_model = model_manager.create_logistic_regression(class_weight='balanced')
    lr_model, lr_metrics = model_manager.train_model(
        lr_model, X_train, y_train, X_val, y_val,
        model_name='Logistic Regression'
    )
    lr_test_metrics = model_manager.evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
    all_results['logistic_regression'] = {**lr_metrics, **lr_test_metrics}

    # Save model
    model_manager.save_model(lr_model, 'logistic_regression')

    # ========== Train Random Forest ==========
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    rf_model = model_manager.create_random_forest(n_estimators=100, class_weight='balanced')
    rf_model, rf_metrics = model_manager.train_model(
        rf_model, X_train, y_train, X_val, y_val,
        model_name='Random Forest'
    )
    rf_test_metrics = model_manager.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    all_results['random_forest'] = {**rf_metrics, **rf_test_metrics}

    # Save model
    model_manager.save_model(rf_model, 'random_forest')

    # ========== Train XGBoost ==========
    print("\n" + "=" * 60)
    print("XGBOOST")
    print("=" * 60)

    xgb_model = model_manager.create_xgboost(scale_pos_weight=scale_pos_weight)
    xgb_model, xgb_metrics = model_manager.train_model(
        xgb_model, X_train, y_train, X_val, y_val,
        model_name='XGBoost'
    )
    xgb_test_metrics = model_manager.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    all_results['xgboost'] = {**xgb_metrics, **xgb_test_metrics}

    # Save model
    model_manager.save_model(xgb_model, 'xgboost')

    # ========== Train Neural Network ==========
    print("\n" + "=" * 60)
    print("NEURAL NETWORK")
    print("=" * 60)

    nn_model = model_manager.create_neural_network(
        input_dim=X_train.shape[1],
        hidden_layers=[64, 32, 16]
    )
    nn_model, nn_metrics = model_manager.train_model(
        nn_model, X_train.values, y_train.values,
        X_val.values, y_val.values,
        model_name='Neural Network',
        use_class_weight=True
    )
    nn_test_metrics = model_manager.evaluate_model(nn_model, X_test.values, y_test.values, 'Neural Network')
    all_results['neural_network'] = {**nn_metrics, **nn_test_metrics}

    # Save model
    model_manager.save_model(nn_model, 'neural_network')

    # ========== Create Ensemble Model ==========
    print("\n" + "=" * 60)
    print("ENSEMBLE MODEL")
    print("=" * 60)

    ensemble_estimators = [
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lr', lr_model)
    ]

    ensemble_model = model_manager.create_ensemble(ensemble_estimators)
    ensemble_model, ensemble_metrics = model_manager.train_model(
        ensemble_model, X_train, y_train, X_val, y_val,
        model_name='Ensemble'
    )
    ensemble_test_metrics = model_manager.evaluate_model(ensemble_model, X_test, y_test, 'Ensemble')
    all_results['ensemble'] = {**ensemble_metrics, **ensemble_test_metrics}

    # Save model
    model_manager.save_model(ensemble_model, 'ensemble')

    # Save all results
    model_manager.results = all_results
    model_manager.save_results()

    return model_manager, all_results

def display_model_comparison(results):
    """
    Display comparison of all models

    Args:
        results: Dictionary of model results
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Test Accuracy': f"{metrics.get('test_accuracy', 0):.4f}",
            'Test ROC-AUC': f"{metrics.get('test_roc_auc', 0):.4f}",
            'Test PR-AUC': f"{metrics.get('test_pr_auc', 0):.4f}",
            'Fraud Precision': f"{metrics.get('test_precision_fraud', 0):.4f}",
            'Fraud Recall': f"{metrics.get('test_recall_fraud', 0):.4f}",
            'Fraud F1-Score': f"{metrics.get('test_f1_fraud', 0):.4f}"
        })

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))

    # Find best model based on F1-score
    best_model = max(results.items(), key=lambda x: x[1].get('test_f1_fraud', 0))
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_model[0].replace('_', ' ').title()}")
    print(f"Test F1-Score (Fraud): {best_model[1]['test_f1_fraud']:.4f}")
    print("=" * 80)

def main():
    """Main training pipeline"""
    print("=" * 80)
    print("FRAUD DETECTION MODEL TRAINING PIPELINE")
    print("=" * 80)

    # Check if processed data exists
    processed_data_path = Path(__file__).parent.parent / 'data' / 'processed_data.pkl'

    if processed_data_path.exists():
        print(f"\nLoading preprocessed data from {processed_data_path}...")
        data_splits = joblib.load(processed_data_path)
        print("✓ Preprocessed data loaded")
    else:
        print("\nPreprocessed data not found. Running preprocessing...")
        preprocessor = FraudDataPreprocessor()
        data_splits = preprocessor.prepare_data()
        preprocessor.save_preprocessor()

        # Save processed data
        joblib.dump(data_splits, processed_data_path)
        print(f"✓ Processed data saved to {processed_data_path}")

    # Train all models
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)

    model_manager, results = train_all_models(data_splits)

    # Display comparison
    display_model_comparison(results)

    # Display feature importance for Random Forest
    try:
        rf_model = model_manager.models['Random Forest']
        feature_columns = data_splits['feature_columns']

        print("\n" + "=" * 80)
        print("TOP 10 MOST IMPORTANT FEATURES (Random Forest)")
        print("=" * 80)

        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]

        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {feature_columns[idx]:20s} - {importances[idx]:.4f}")

    except Exception as e:
        print(f"Could not display feature importances: {e}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nTrained models saved to: models/")
    print("Results saved to: results/model_results.json")
    print("\nNext steps:")
    print("  1. Run visualizations: python src/visualizations.py")
    print("  2. Make predictions: python src/predict.py")

if __name__ == "__main__":
    main()