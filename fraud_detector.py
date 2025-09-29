"""
ML Fraud Detection System
A comprehensive machine learning system for detecting fraudulent transactions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

class FraudDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

    def load_data(self, filepath):
        """Load and preprocess transaction data"""
        self.data = pd.read_csv(filepath)
        return self.data

    def preprocess_data(self, data):
        """Feature engineering and data preprocessing"""
        # Add your preprocessing logic here
        processed_data = data.copy()

        # Example preprocessing steps
        processed_data['transaction_hour'] = pd.to_datetime(processed_data['timestamp']).dt.hour
        processed_data['amount_log'] = np.log1p(processed_data['amount'])

        return processed_data

    def train_model(self, X, y):
        """Train the fraud detection model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        return accuracy

    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save_model(self, filepath):
        """Save trained model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)

    def load_model(self, filepath):
        """Load trained model from disk"""
        saved_objects = joblib.load(filepath)
        self.model = saved_objects['model']
        self.scaler = saved_objects['scaler']
        self.feature_columns = saved_objects['feature_columns']

if __name__ == "__main__":
    # Example usage
    detector = FraudDetector()
    print("ML Fraud Detection System initialized")
    print("Ready to detect fraudulent transactions!")