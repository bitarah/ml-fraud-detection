"""
Visualization Generator for Fraud Detection Models
Creates comprehensive charts and plots for model evaluation
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    auc, classification_report
)
from tensorflow import keras

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class FraudVisualizationGenerator:
    """
    Generate visualizations for fraud detection models
    """

    def __init__(self, output_dir=None):
        """
        Initialize visualization generator

        Args:
            output_dir: Directory to save plots
        """
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent / 'results' / 'plots'
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Also save to screenshots for GitHub
        self.screenshots_dir = Path(__file__).parent.parent / 'screenshots'
        self.screenshots_dir.mkdir(exist_ok=True, parents=True)

    def plot_class_distribution(self, y_train, y_val, y_test):
        """
        Plot class distribution across datasets

        Args:
            y_train, y_val, y_test: Target arrays
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        datasets = [
            ('Training Set', y_train),
            ('Validation Set', y_val),
            ('Test Set', y_test)
        ]

        for ax, (title, y) in zip(axes, datasets):
            counts = y.value_counts()
            colors = ['#2ecc71', '#e74c3c']

            ax.bar(['Legitimate', 'Fraud'], counts.values, color=colors, alpha=0.7)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Count')

            # Add percentage labels
            total = len(y)
            for i, count in enumerate(counts.values):
                percentage = (count / total) * 100
                ax.text(i, count, f'{count:,}\n({percentage:.2f}%)',
                       ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        self._save_plot('class_distribution.png')
        print("✓ Class distribution plot saved")

    def plot_confusion_matrices(self, models, X_test, y_test):
        """
        Plot confusion matrices for all models

        Args:
            models: Dictionary of trained models
            X_test, y_test: Test data
        """
        n_models = len(models)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (model_name, model) in enumerate(models.items()):
            if idx >= len(axes):
                break

            # Get predictions
            is_keras = isinstance(model, keras.Model)
            if is_keras:
                y_pred = (model.predict(X_test.values, verbose=0) > 0.5).astype(int).flatten()
            else:
                y_pred = model.predict(X_test)

            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False, square=True)
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_xticklabels(['Legitimate', 'Fraud'])
            axes[idx].set_yticklabels(['Legitimate', 'Fraud'])

        # Hide extra subplots
        for idx in range(len(models), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        self._save_plot('confusion_matrices.png')
        print("✓ Confusion matrices plot saved")

    def plot_roc_curves(self, models, X_test, y_test):
        """
        Plot ROC curves for all models

        Args:
            models: Dictionary of trained models
            X_test, y_test: Test data
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        for idx, (model_name, model) in enumerate(models.items()):
            # Get prediction probabilities
            is_keras = isinstance(model, keras.Model)
            if is_keras:
                y_proba = model.predict(X_test.values, verbose=0).flatten()
            else:
                y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            # Plot
            color = colors[idx % len(colors)]
            ax.plot(fpr, tpr, color=color, lw=2, alpha=0.8,
                   label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3, label='Random Classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        self._save_plot('roc_curves.png')
        print("✓ ROC curves plot saved")

    def plot_precision_recall_curves(self, models, X_test, y_test):
        """
        Plot Precision-Recall curves for all models

        Args:
            models: Dictionary of trained models
            X_test, y_test: Test data
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        for idx, (model_name, model) in enumerate(models.items()):
            # Get prediction probabilities
            is_keras = isinstance(model, keras.Model)
            if is_keras:
                y_proba = model.predict(X_test.values, verbose=0).flatten()
            else:
                y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)

            # Plot
            color = colors[idx % len(colors)]
            ax.plot(recall, precision, color=color, lw=2, alpha=0.8,
                   label=f'{model_name.replace("_", " ").title()} (AUC = {pr_auc:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        self._save_plot('precision_recall_curves.png')
        print("✓ Precision-Recall curves plot saved")

    def plot_model_comparison(self, results):
        """
        Plot model comparison metrics

        Args:
            results: Dictionary of model results
        """
        # Prepare data
        metrics_to_plot = [
            ('test_accuracy', 'Accuracy'),
            ('test_roc_auc', 'ROC-AUC'),
            ('test_pr_auc', 'PR-AUC'),
            ('test_f1_fraud', 'F1-Score (Fraud)')
        ]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
            ax = axes[idx]

            # Extract metric values
            model_names = []
            values = []

            for model_name, metrics in results.items():
                model_names.append(model_name.replace('_', ' ').title())
                values.append(metrics.get(metric_key, 0))

            # Create bar plot
            bars = ax.bar(range(len(model_names)), values,
                         color=[colors[i % len(colors)] for i in range(len(model_names))],
                         alpha=0.7)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{value:.4f}',
                       ha='center', va='bottom', fontweight='bold')

            ax.set_xlabel('Model', fontsize=11)
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylim([0, max(values) * 1.15])
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save_plot('model_comparison.png')
        print("✓ Model comparison plot saved")

    def plot_feature_importance(self, model, feature_columns, top_n=15):
        """
        Plot feature importance for tree-based models

        Args:
            model: Trained model with feature_importances_ attribute
            feature_columns: List of feature names
            top_n: Number of top features to display
        """
        if not hasattr(model, 'feature_importances_'):
            print("✗ Model does not have feature importances")
            return

        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, top_n))

        bars = ax.barh(range(top_n), importances[indices], color=colors, alpha=0.7)

        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_columns[i] for i in indices])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2.,
                   f'{importances[indices[i]]:.4f}',
                   ha='left', va='center', fontweight='bold', fontsize=9)

        plt.tight_layout()
        self._save_plot('feature_importance.png')
        print("✓ Feature importance plot saved")

    def create_summary_dashboard(self, results):
        """
        Create a summary dashboard with key metrics

        Args:
            results: Dictionary of model results
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('Fraud Detection Model Performance Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1].get('test_f1_fraud', 0))
        best_model_name = best_model[0].replace('_', ' ').title()

        # 1. Best Model Metrics (large text)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        metrics_text = (
            f"Best Model: {best_model_name}\n\n"
            f"Test Accuracy: {best_model[1]['test_accuracy']:.4f}  |  "
            f"ROC-AUC: {best_model[1]['test_roc_auc']:.4f}  |  "
            f"PR-AUC: {best_model[1]['test_pr_auc']:.4f}\n"
            f"Fraud Detection - Precision: {best_model[1]['test_precision_fraud']:.4f}  |  "
            f"Recall: {best_model[1]['test_recall_fraud']:.4f}  |  "
            f"F1-Score: {best_model[1]['test_f1_fraud']:.4f}"
        )
        ax1.text(0.5, 0.5, metrics_text, ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 2. F1-Score comparison
        ax2 = fig.add_subplot(gs[1, 0])
        model_names = [name.replace('_', ' ').title() for name in results.keys()]
        f1_scores = [metrics['test_f1_fraud'] for metrics in results.values()]
        colors_bar = ['#e74c3c' if name == best_model_name else '#3498db' for name in model_names]
        ax2.barh(model_names, f1_scores, color=colors_bar, alpha=0.7)
        ax2.set_xlabel('F1-Score (Fraud)')
        ax2.set_title('F1-Score Comparison', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # 3. ROC-AUC comparison
        ax3 = fig.add_subplot(gs[1, 1])
        roc_aucs = [metrics['test_roc_auc'] for metrics in results.values()]
        ax3.barh(model_names, roc_aucs, color=colors_bar, alpha=0.7)
        ax3.set_xlabel('ROC-AUC')
        ax3.set_title('ROC-AUC Comparison', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        # 4. Precision vs Recall
        ax4 = fig.add_subplot(gs[1, 2])
        precisions = [metrics['test_precision_fraud'] for metrics in results.values()]
        recalls = [metrics['test_recall_fraud'] for metrics in results.values()]
        ax4.scatter(recalls, precisions, s=200, alpha=0.6, c=range(len(model_names)), cmap='viridis')
        for i, name in enumerate(model_names):
            ax4.annotate(name, (recalls[i], precisions[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision vs Recall', fontweight='bold')
        ax4.grid(alpha=0.3)

        # 5. Best Model Confusion Matrix
        ax5 = fig.add_subplot(gs[2, :])
        cm = np.array(best_model[1]['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'], cbar=False)
        ax5.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
        ax5.set_ylabel('Actual')
        ax5.set_xlabel('Predicted')

        plt.tight_layout()
        self._save_plot('summary_dashboard.png')
        # Also save to screenshots as main demo image
        plt.savefig(self.screenshots_dir / 'demo.png', dpi=150, bbox_inches='tight')
        print("✓ Summary dashboard saved")

    def _save_plot(self, filename):
        """Save plot to both results and screenshots directories"""
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.savefig(self.screenshots_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main function to generate all visualizations"""
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Load results
    results_path = Path(__file__).parent.parent / 'results' / 'model_results.json'
    if not results_path.exists():
        print("✗ Model results not found. Please run training first:")
        print("  python src/train.py")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    print(f"✓ Loaded results for {len(results)} models")

    # Load data and models
    data_path = Path(__file__).parent.parent / 'data' / 'processed_data.pkl'
    models_dir = Path(__file__).parent.parent / 'models'

    if not data_path.exists():
        print("✗ Processed data not found. Please run training first.")
        return

    data_splits = joblib.load(data_path)
    print("✓ Loaded processed data")

    # Load models
    models = {}
    for model_name in results.keys():
        pkl_path = models_dir / f"{model_name}.pkl"
        h5_path = models_dir / f"{model_name}.h5"

        if pkl_path.exists():
            models[model_name] = joblib.load(pkl_path)
        elif h5_path.exists():
            models[model_name] = keras.models.load_model(h5_path)

    print(f"✓ Loaded {len(models)} models")

    # Initialize visualizer
    viz = FraudVisualizationGenerator()

    # Generate plots
    print("\nGenerating plots...")

    # 1. Class distribution
    viz.plot_class_distribution(
        data_splits['y_train'],
        data_splits['y_val'],
        data_splits['y_test']
    )

    # 2. Confusion matrices
    viz.plot_confusion_matrices(models, data_splits['X_test'], data_splits['y_test'])

    # 3. ROC curves
    viz.plot_roc_curves(models, data_splits['X_test'], data_splits['y_test'])

    # 4. Precision-Recall curves
    viz.plot_precision_recall_curves(models, data_splits['X_test'], data_splits['y_test'])

    # 5. Model comparison
    viz.plot_model_comparison(results)

    # 6. Feature importance (for Random Forest)
    if 'random_forest' in models:
        viz.plot_feature_importance(
            models['random_forest'],
            data_splits['feature_columns']
        )

    # 7. Summary dashboard
    viz.create_summary_dashboard(results)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nPlots saved to:")
    print(f"  - {viz.output_dir}")
    print(f"  - {viz.screenshots_dir}")

if __name__ == "__main__":
    main()