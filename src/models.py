"""
Model Architectures for Fraud Detection
Implements multiple ML models for comparison
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, roc_curve
)
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib
from pathlib import Path
import json

class FraudDetectionModels:
    """
    Collection of fraud detection models
    """

    def __init__(self, random_state=42):
        """
        Initialize models

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def create_logistic_regression(self, class_weight='balanced'):
        """
        Create Logistic Regression model

        Args:
            class_weight: Strategy for handling class imbalance

        Returns:
            Logistic Regression model
        """
        model = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        return model

    def create_random_forest(self, n_estimators=100, class_weight='balanced'):
        """
        Create Random Forest model

        Args:
            n_estimators: Number of trees
            class_weight: Strategy for handling class imbalance

        Returns:
            Random Forest model
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        return model

    def create_xgboost(self, scale_pos_weight=None):
        """
        Create XGBoost model

        Args:
            scale_pos_weight: Weight for positive class (fraud)

        Returns:
            XGBoost model
        """
        if scale_pos_weight is None:
            scale_pos_weight = 1

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        return model

    def create_neural_network(self, input_dim, hidden_layers=[64, 32, 16]):
        """
        Create Neural Network model with Keras

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes

        Returns:
            Keras Sequential model
        """
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization()
        ])

        # Add hidden layers with dropout
        for units in hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))

        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

        return model

    def create_ensemble(self, estimators):
        """
        Create ensemble voting classifier

        Args:
            estimators: List of (name, model) tuples

        Returns:
            Voting Classifier
        """
        model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        return model

    def train_model(self, model, X_train, y_train, X_val=None, y_val=None,
                   model_name='model', use_class_weight=False):
        """
        Train a model and evaluate on validation set

        Args:
            model: Model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_name: Name for the model
            use_class_weight: Whether to use class weights (for NN)

        Returns:
            Trained model and metrics
        """
        print(f"\nTraining {model_name}...")

        # Check if it's a Keras model
        is_keras = isinstance(model, keras.Model)

        if is_keras:
            # Calculate class weights for neural network
            if use_class_weight:
                from sklearn.utils.class_weight import compute_class_weight
                class_weights = compute_class_weight(
                    'balanced',
                    classes=np.unique(y_train),
                    y=y_train
                )
                class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
            else:
                class_weight_dict = None

            # Early stopping callback
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )

            # Train neural network
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=50,
                batch_size=512,
                class_weight=class_weight_dict,
                callbacks=[early_stop],
                verbose=0
            )

            # Get predictions
            y_train_pred = (model.predict(X_train, verbose=0) > 0.5).astype(int).flatten()
            if X_val is not None:
                y_val_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
                y_val_proba = model.predict(X_val, verbose=0).flatten()
        else:
            # Train sklearn model
            model.fit(X_train, y_train)

            # Get predictions
            y_train_pred = model.predict(X_train)
            if X_val is not None:
                y_val_pred = model.predict(X_val)
                y_val_proba = model.predict_proba(X_val)[:, 1]

        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'train_accuracy': np.mean(y_train_pred == y_train)
        }

        if X_val is not None:
            metrics['val_accuracy'] = np.mean(y_val_pred == y_val)
            metrics['val_roc_auc'] = roc_auc_score(y_val, y_val_proba)

            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
            metrics['val_pr_auc'] = auc(recall, precision)

            # Classification report
            report = classification_report(y_val, y_val_pred, output_dict=True)
            metrics['val_precision_fraud'] = report['1']['precision']
            metrics['val_recall_fraud'] = report['1']['recall']
            metrics['val_f1_fraud'] = report['1']['f1-score']

            print(f"✓ {model_name} trained successfully")
            print(f"  Validation Accuracy: {metrics['val_accuracy']:.4f}")
            print(f"  Validation ROC-AUC:  {metrics['val_roc_auc']:.4f}")
            print(f"  Validation PR-AUC:   {metrics['val_pr_auc']:.4f}")
            print(f"  Fraud Precision:     {metrics['val_precision_fraud']:.4f}")
            print(f"  Fraud Recall:        {metrics['val_recall_fraud']:.4f}")
            print(f"  Fraud F1-Score:      {metrics['val_f1_fraud']:.4f}")

        self.models[model_name] = model
        self.results[model_name] = metrics

        return model, metrics

    def evaluate_model(self, model, X_test, y_test, model_name='model'):
        """
        Evaluate model on test set

        Args:
            model: Trained model
            X_test, y_test: Test data
            model_name: Name of the model

        Returns:
            Dictionary of test metrics
        """
        print(f"\nEvaluating {model_name} on test set...")

        # Check if it's a Keras model
        is_keras = isinstance(model, keras.Model)

        if is_keras:
            y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
            y_proba = model.predict(X_test, verbose=0).flatten()
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'test_accuracy': np.mean(y_pred == y_test),
            'test_roc_auc': roc_auc_score(y_test, y_proba)
        }

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        metrics['test_pr_auc'] = auc(recall, precision)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['test_precision_fraud'] = report['1']['precision']
        metrics['test_recall_fraud'] = report['1']['recall']
        metrics['test_f1_fraud'] = report['1']['f1-score']

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        print(f"✓ {model_name} Test Results:")
        print(f"  Test Accuracy:       {metrics['test_accuracy']:.4f}")
        print(f"  Test ROC-AUC:        {metrics['test_roc_auc']:.4f}")
        print(f"  Test PR-AUC:         {metrics['test_pr_auc']:.4f}")
        print(f"  Fraud Precision:     {metrics['test_precision_fraud']:.4f}")
        print(f"  Fraud Recall:        {metrics['test_recall_fraud']:.4f}")
        print(f"  Fraud F1-Score:      {metrics['test_f1_fraud']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0][0]:6d}  FP: {cm[0][1]:6d}")
        print(f"  FN: {cm[1][0]:6d}  TP: {cm[1][1]:6d}")

        return metrics

    def save_model(self, model, model_name, output_dir=None):
        """
        Save trained model to disk

        Args:
            model: Trained model
            model_name: Name of the model
            output_dir: Directory to save model
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'models'

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Check if it's a Keras model
        is_keras = isinstance(model, keras.Model)

        if is_keras:
            model_path = output_dir / f"{model_name}.h5"
            model.save(model_path)
        else:
            model_path = output_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)

        print(f"✓ Model saved to {model_path}")

    def save_results(self, output_dir=None):
        """
        Save all model results to JSON

        Args:
            output_dir: Directory to save results
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'results'

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        results_path = output_dir / 'model_results.json'

        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"✓ Results saved to {results_path}")

def main():
    """Test model creation"""
    print("=" * 60)
    print("Fraud Detection Models")
    print("=" * 60)

    models = FraudDetectionModels()

    # Test model creation
    print("\nTesting model creation...")
    lr = models.create_logistic_regression()
    print("✓ Logistic Regression created")

    rf = models.create_random_forest()
    print("✓ Random Forest created")

    xgb_model = models.create_xgboost()
    print("✓ XGBoost created")

    nn = models.create_neural_network(input_dim=35)
    print("✓ Neural Network created")
    print(f"  Total parameters: {nn.count_params():,}")

    print("\nAll models created successfully!")

if __name__ == "__main__":
    main()