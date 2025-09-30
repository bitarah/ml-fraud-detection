"""
Prediction Script for Fraud Detection
Make predictions on new transaction data
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import FraudDataPreprocessor

class FraudPredictor:
    """
    Load trained models and make predictions
    """

    def __init__(self, model_name='xgboost', models_dir=None):
        """
        Initialize predictor

        Args:
            model_name: Name of the model to use
            models_dir: Directory containing saved models
        """
        if models_dir is None:
            self.models_dir = Path(__file__).parent.parent / 'models'
        else:
            self.models_dir = Path(models_dir)

        self.model_name = model_name
        self.model = None
        self.preprocessor = None
        self.feature_columns = None

    def load_model(self):
        """Load trained model and preprocessor"""
        # Load model
        pkl_path = self.models_dir / f"{self.model_name}.pkl"
        h5_path = self.models_dir / f"{self.model_name}.h5"

        if pkl_path.exists():
            self.model = joblib.load(pkl_path)
            print(f"✓ Loaded model from {pkl_path}")
        elif h5_path.exists():
            self.model = keras.models.load_model(h5_path)
            print(f"✓ Loaded model from {h5_path}")
        else:
            raise FileNotFoundError(
                f"Model '{self.model_name}' not found in {self.models_dir}\n"
                "Available models: logistic_regression, random_forest, xgboost, neural_network, ensemble"
            )

        # Load preprocessor
        preprocessor_path = self.models_dir / 'preprocessor.pkl'
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {preprocessor_path}\n"
                "Please run training first: python src/train.py"
            )

        preprocessor_data = FraudDataPreprocessor.load_preprocessor(preprocessor_path)
        self.scaler = preprocessor_data['scaler']
        self.amount_scaler = preprocessor_data['amount_scaler']
        self.feature_columns = preprocessor_data['feature_columns']

        print(f"✓ Loaded preprocessor with {len(self.feature_columns)} features")

    def preprocess_transaction(self, transaction_data):
        """
        Preprocess a single transaction or batch of transactions

        Args:
            transaction_data: DataFrame with transaction data

        Returns:
            Preprocessed features ready for prediction
        """
        df = transaction_data.copy()

        # Engineer features (same as training)
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Day'] = (df['Time'] / 86400).astype(int)
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)

        # Amount categories (using simple quartile-based binning)
        df['Amount_Category'] = pd.cut(
            df['Amount'],
            bins=[-np.inf, 50, 100, 200, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # Select only the features used during training
        X = df[self.feature_columns]

        # Scale features
        amount_cols = ['Amount', 'Amount_log']
        other_cols = [col for col in X.columns if col not in amount_cols]

        amount_cols_present = [col for col in amount_cols if col in X.columns]

        if amount_cols_present:
            X_amount = self.amount_scaler.transform(X[amount_cols_present])
            X_other = self.scaler.transform(X[other_cols])
            X_scaled = np.concatenate([X_other, X_amount], axis=1)
            cols_order = other_cols + amount_cols_present
        else:
            X_scaled = self.scaler.transform(X)
            cols_order = X.columns.tolist()

        X_scaled = pd.DataFrame(X_scaled, columns=cols_order, index=X.index)

        return X_scaled

    def predict(self, transaction_data, return_proba=True):
        """
        Make fraud predictions on transaction data

        Args:
            transaction_data: DataFrame with transaction data
            return_proba: Whether to return probabilities

        Returns:
            Predictions (and probabilities if return_proba=True)
        """
        if self.model is None:
            self.load_model()

        # Preprocess
        X = self.preprocess_transaction(transaction_data)

        # Check if Keras model
        is_keras = isinstance(self.model, keras.Model)

        if is_keras:
            predictions = (self.model.predict(X.values, verbose=0) > 0.5).astype(int).flatten()
            if return_proba:
                probabilities = self.model.predict(X.values, verbose=0).flatten()
        else:
            predictions = self.model.predict(X)
            if return_proba:
                probabilities = self.model.predict_proba(X)[:, 1]

        if return_proba:
            return predictions, probabilities
        else:
            return predictions

    def predict_single(self, transaction_dict):
        """
        Make prediction on a single transaction

        Args:
            transaction_dict: Dictionary with transaction features

        Returns:
            Tuple of (prediction, probability, confidence)
        """
        # Convert to DataFrame
        df = pd.DataFrame([transaction_dict])

        # Make prediction
        prediction, probability = self.predict(df, return_proba=True)

        # Calculate confidence
        confidence = probability[0] if prediction[0] == 1 else 1 - probability[0]

        result = {
            'is_fraud': bool(prediction[0]),
            'fraud_probability': float(probability[0]),
            'confidence': float(confidence),
            'risk_level': self._get_risk_level(probability[0])
        }

        return result

    def _get_risk_level(self, probability):
        """Categorize risk level based on fraud probability"""
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.6:
            return 'MEDIUM'
        elif probability < 0.85:
            return 'HIGH'
        else:
            return 'CRITICAL'

def demo_prediction():
    """Demonstrate prediction on sample transactions"""
    print("=" * 80)
    print("FRAUD DETECTION - PREDICTION DEMO")
    print("=" * 80)

    # Initialize predictor
    predictor = FraudPredictor(model_name='xgboost')
    predictor.load_model()

    # Sample legitimate transaction
    legitimate_transaction = {
        'Time': 12345,
        'V1': -1.359807,
        'V2': -0.072781,
        'V3': 2.536347,
        'V4': 1.378155,
        'V5': -0.338321,
        'V6': 0.462388,
        'V7': 0.239599,
        'V8': 0.098698,
        'V9': 0.363787,
        'V10': 0.090794,
        'V11': -0.551600,
        'V12': -0.617801,
        'V13': -0.991390,
        'V14': -0.311169,
        'V15': 1.468177,
        'V16': -0.470401,
        'V17': 0.207971,
        'V18': 0.025791,
        'V19': 0.403993,
        'V20': 0.251412,
        'V21': -0.018307,
        'V22': 0.277838,
        'V23': -0.110474,
        'V24': 0.066928,
        'V25': 0.128539,
        'V26': -0.189115,
        'V27': 0.133558,
        'V28': -0.021053,
        'Amount': 149.62
    }

    # Sample suspicious transaction
    suspicious_transaction = {
        'Time': 54321,
        'V1': 2.5,
        'V2': 3.1,
        'V3': -5.2,
        'V4': 4.8,
        'V5': -2.9,
        'V6': 1.5,
        'V7': -3.2,
        'V8': 2.1,
        'V9': -1.8,
        'V10': 3.5,
        'V11': -4.2,
        'V12': 2.7,
        'V13': -3.9,
        'V14': 5.1,
        'V15': -2.3,
        'V16': 3.8,
        'V17': -4.5,
        'V18': 2.9,
        'V19': -3.1,
        'V20': 4.2,
        'V21': -2.8,
        'V22': 3.3,
        'V23': -4.1,
        'V24': 2.4,
        'V25': -3.6,
        'V26': 4.7,
        'V27': -2.1,
        'V28': 3.9,
        'Amount': 1250.00
    }

    print("\n" + "=" * 80)
    print("SAMPLE TRANSACTION 1 (Expected: Legitimate)")
    print("=" * 80)
    result1 = predictor.predict_single(legitimate_transaction)
    print(f"Prediction:         {'FRAUD' if result1['is_fraud'] else 'LEGITIMATE'}")
    print(f"Fraud Probability:  {result1['fraud_probability']:.4f}")
    print(f"Confidence:         {result1['confidence']:.4f}")
    print(f"Risk Level:         {result1['risk_level']}")

    print("\n" + "=" * 80)
    print("SAMPLE TRANSACTION 2 (Expected: Suspicious)")
    print("=" * 80)
    result2 = predictor.predict_single(suspicious_transaction)
    print(f"Prediction:         {'FRAUD' if result2['is_fraud'] else 'LEGITIMATE'}")
    print(f"Fraud Probability:  {result2['fraud_probability']:.4f}")
    print(f"Confidence:         {result2['confidence']:.4f}")
    print(f"Risk Level:         {result2['risk_level']}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)

def main():
    """Main prediction function"""
    demo_prediction()

if __name__ == "__main__":
    main()