"""
Data Download Script for Credit Card Fraud Detection
Downloads the dataset from Kaggle or uses a synthetic fallback
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def download_kaggle_dataset():
    """
    Download Credit Card Fraud Detection dataset from Kaggle
    Requires kaggle API credentials in ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
        print("Attempting to download dataset from Kaggle...")

        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent.parent / 'data'
        data_dir.mkdir(exist_ok=True)

        # Download dataset
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path=str(data_dir),
            unzip=True
        )

        print(f"✓ Dataset downloaded successfully to {data_dir}")
        return True
    except Exception as e:
        print(f"✗ Could not download from Kaggle: {e}")
        print("  To use Kaggle API: https://www.kaggle.com/docs/api")
        return False

def generate_synthetic_data():
    """
    Generate synthetic fraud detection dataset for testing
    Mimics the structure of the Credit Card Fraud Detection dataset
    """
    print("Generating synthetic dataset for testing...")

    np.random.seed(42)
    n_samples = 50000
    fraud_rate = 0.002  # 0.2% fraud rate
    n_frauds = int(n_samples * fraud_rate)

    # Generate features (simulating PCA-transformed features)
    data = {
        'Time': np.random.randint(0, 172800, n_samples),  # 48 hours in seconds
        'Amount': np.random.lognormal(3.5, 2, n_samples)  # Log-normal distribution for amounts
    }

    # Generate V1-V28 features (simulating PCA components)
    for i in range(1, 29):
        if i % 3 == 0:
            # Some features have different distributions for fraud
            legitimate = np.random.normal(0, 1, n_samples - n_frauds)
            fraudulent = np.random.normal(2, 1.5, n_frauds)
            data[f'V{i}'] = np.concatenate([legitimate, fraudulent])
        else:
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)

    # Generate target (0 = legitimate, 1 = fraud)
    data['Class'] = np.concatenate([
        np.zeros(n_samples - n_frauds, dtype=int),
        np.ones(n_frauds, dtype=int)
    ])

    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / 'creditcard.csv'
    df.to_csv(output_path, index=False)

    print(f"✓ Synthetic dataset generated with {n_samples:,} transactions")
    print(f"  - Legitimate transactions: {n_samples - n_frauds:,}")
    print(f"  - Fraudulent transactions: {n_frauds:,}")
    print(f"  - Saved to: {output_path}")

    return True

def verify_dataset():
    """Verify that the dataset exists and is valid"""
    data_path = Path(__file__).parent.parent / 'data' / 'creditcard.csv'

    if not data_path.exists():
        return False

    try:
        df = pd.read_csv(data_path, nrows=5)
        required_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]

        if all(col in df.columns for col in required_columns):
            print(f"✓ Dataset verified at {data_path}")
            return True
        else:
            print("✗ Dataset is missing required columns")
            return False
    except Exception as e:
        print(f"✗ Error reading dataset: {e}")
        return False

def main():
    """Main function to download or generate dataset"""
    print("=" * 60)
    print("Credit Card Fraud Detection - Data Setup")
    print("=" * 60)

    # Check if dataset already exists
    if verify_dataset():
        print("\n✓ Dataset already exists and is valid!")
        return True

    # Try to download from Kaggle
    print("\nAttempting to download real dataset from Kaggle...")
    if download_kaggle_dataset() and verify_dataset():
        return True

    # Fallback to synthetic data
    print("\nFalling back to synthetic dataset generation...")
    if generate_synthetic_data() and verify_dataset():
        print("\n" + "=" * 60)
        print("NOTE: Using synthetic data for demonstration.")
        print("For real results, download from Kaggle:")
        print("https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print("=" * 60)
        return True

    print("\n✗ Failed to setup dataset")
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)