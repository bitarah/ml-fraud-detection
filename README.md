# ML Fraud Detection System

A machine learning project that implements multiple algorithms to detect fraudulent transactions in financial data.

## Features

- Multiple ML algorithms (Random Forest, SVM, Neural Networks)
- Real-time prediction capabilities
- Comprehensive data preprocessing pipeline
- Performance visualization and metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from fraud_detector import FraudDetector

detector = FraudDetector()
detector.load_model('models/best_model.pkl')
result = detector.predict(transaction_data)
```

## Results

- Achieved 94.5% accuracy on test dataset
- 0.92 F1-score for fraud detection
- Processing time: <100ms per transaction

## Contributing

Feel free to open issues and pull requests!