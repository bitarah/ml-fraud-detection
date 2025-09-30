# Data Directory

This directory contains the fraud detection dataset.

## Contents

- `creditcard.csv` - Raw credit card transaction data (gitignored)
- `processed_data.pkl` - Preprocessed and split data for training (gitignored)

## Dataset

The Credit Card Fraud Detection dataset contains:
- **284,807 transactions** from European cardholders in September 2013
- **492 fraudulent transactions** (0.172% of all transactions)
- **30 features**: Time, Amount, and V1-V28 (PCA-transformed features)

### Features

- **Time**: Seconds elapsed between this transaction and the first transaction
- **V1-V28**: PCA-transformed features (anonymized for confidentiality)
- **Amount**: Transaction amount
- **Class**: Target variable (0 = legitimate, 1 = fraud)

## Download

Run the download script to fetch the dataset:

```bash
python src/download_data.py
```

This will:
1. Attempt to download from Kaggle API
2. Fall back to generating synthetic data if Kaggle is unavailable

## Source

- **Original Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Citation**: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015