"""
Data Preprocessing Pipeline for Fraud Detection
Handles feature engineering, scaling, and data splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import json

class FraudDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for fraud detection
    """

    def __init__(self, data_path=None):
        """
        Initialize the preprocessor

        Args:
            data_path: Path to the creditcard.csv file
        """
        if data_path is None:
            self.data_path = Path(__file__).parent.parent / 'data' / 'creditcard.csv'
        else:
            self.data_path = Path(data_path)

        self.scaler = StandardScaler()
        self.amount_scaler = RobustScaler()
        self.feature_columns = None
        self.data = None

    def load_data(self):
        """Load the credit card transaction data"""
        print(f"Loading data from {self.data_path}...")

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found at {self.data_path}\n"
                "Please run 'python src/download_data.py' first"
            )

        self.data = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.data):,} transactions")

        # Display class distribution
        fraud_count = self.data['Class'].sum()
        legitimate_count = len(self.data) - fraud_count
        fraud_percentage = (fraud_count / len(self.data)) * 100

        print(f"  - Legitimate: {legitimate_count:,} ({100-fraud_percentage:.3f}%)")
        print(f"  - Fraudulent: {fraud_count:,} ({fraud_percentage:.3f}%)")

        return self.data

    def engineer_features(self, df):
        """
        Create additional features from existing data

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Time-based features
        df['Hour'] = (df['Time'] / 3600) % 24  # Hour of day
        df['Day'] = (df['Time'] / 86400).astype(int)  # Day number

        # Transaction amount features
        df['Amount_log'] = np.log1p(df['Amount'])  # Log transform amount

        # Binned time features (day vs night transactions)
        df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)

        # Amount categories (based on quartiles)
        amount_quartiles = df['Amount'].quantile([0.25, 0.5, 0.75])
        df['Amount_Category'] = pd.cut(
            df['Amount'],
            bins=[-np.inf, amount_quartiles[0.25], amount_quartiles[0.5],
                  amount_quartiles[0.75], np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)

        print(f"✓ Engineered {len(df.columns) - len(self.data.columns)} new features")

        return df

    def scale_features(self, X_train, X_val, X_test):
        """
        Scale features using StandardScaler

        Args:
            X_train, X_val, X_test: Feature matrices

        Returns:
            Scaled feature matrices
        """
        # Identify amount-related columns for robust scaling
        amount_cols = ['Amount', 'Amount_log']
        other_cols = [col for col in X_train.columns if col not in amount_cols]

        # Scale amount features with RobustScaler (resistant to outliers)
        if any(col in X_train.columns for col in amount_cols):
            amount_cols_present = [col for col in amount_cols if col in X_train.columns]

            X_train_amount = self.amount_scaler.fit_transform(X_train[amount_cols_present])
            X_val_amount = self.amount_scaler.transform(X_val[amount_cols_present])
            X_test_amount = self.amount_scaler.transform(X_test[amount_cols_present])

            # Scale other features with StandardScaler
            X_train_other = self.scaler.fit_transform(X_train[other_cols])
            X_val_other = self.scaler.transform(X_val[other_cols])
            X_test_other = self.scaler.transform(X_test[other_cols])

            # Combine scaled features
            X_train_scaled = np.concatenate([X_train_other, X_train_amount], axis=1)
            X_val_scaled = np.concatenate([X_val_other, X_val_amount], axis=1)
            X_test_scaled = np.concatenate([X_test_other, X_test_amount], axis=1)

            # Reconstruct column order
            cols_order = other_cols + amount_cols_present
        else:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            cols_order = X_train.columns.tolist()

        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=cols_order, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=cols_order, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=cols_order, index=X_test.index)

        print("✓ Features scaled successfully")

        return X_train_scaled, X_val_scaled, X_test_scaled

    def prepare_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Prepare data for model training

        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing train, validation, and test sets
        """
        if self.data is None:
            self.load_data()

        # Engineer features
        df_processed = self.engineer_features(self.data)

        # Separate features and target
        X = df_processed.drop(['Class'], axis=1)
        y = df_processed['Class']

        # Store feature columns
        self.feature_columns = X.columns.tolist()

        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=random_state,
            stratify=y_temp
        )

        print(f"\n✓ Data split completed:")
        print(f"  - Training set:   {len(X_train):,} samples ({y_train.sum()} frauds)")
        print(f"  - Validation set: {len(X_val):,} samples ({y_val.sum()} frauds)")
        print(f"  - Test set:       {len(X_test):,} samples ({y_test.sum()} frauds)")

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )

        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_columns': self.feature_columns
        }

    def save_preprocessor(self, output_dir=None):
        """
        Save the preprocessor (scalers and feature columns)

        Args:
            output_dir: Directory to save preprocessor
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'models'

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        preprocessor_path = output_dir / 'preprocessor.pkl'

        joblib.dump({
            'scaler': self.scaler,
            'amount_scaler': self.amount_scaler,
            'feature_columns': self.feature_columns
        }, preprocessor_path)

        print(f"✓ Preprocessor saved to {preprocessor_path}")

    @staticmethod
    def load_preprocessor(preprocessor_path):
        """
        Load a saved preprocessor

        Args:
            preprocessor_path: Path to saved preprocessor

        Returns:
            Dictionary containing scaler and feature columns
        """
        return joblib.load(preprocessor_path)

def main():
    """Main function for testing the preprocessing pipeline"""
    print("=" * 60)
    print("Fraud Detection Data Preprocessing Pipeline")
    print("=" * 60)

    # Initialize preprocessor
    preprocessor = FraudDataPreprocessor()

    # Load and prepare data
    data_splits = preprocessor.prepare_data()

    # Save preprocessor
    preprocessor.save_preprocessor()

    # Save processed data for training
    data_dir = Path(__file__).parent.parent / 'data'
    output_path = data_dir / 'processed_data.pkl'
    joblib.dump(data_splits, output_path)
    print(f"✓ Processed data saved to {output_path}")

    # Display summary statistics
    print("\n" + "=" * 60)
    print("Data Preprocessing Summary")
    print("=" * 60)
    print(f"Total features: {len(data_splits['feature_columns'])}")
    print(f"Feature list: {', '.join(data_splits['feature_columns'][:10])}...")
    print("\nPreprocessing complete! Ready for model training.")

if __name__ == "__main__":
    main()