#!/usr/bin/env python3
"""
Model Evaluation Script for Nifty Trading Agent
Analyzes model calibration and creates detailed performance reports
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sklearn.metrics import brier_score_loss

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import load_features_for_training
from utils.io_utils import load_yaml_config

logger = get_logger(__name__)

class ModelEvaluator:
    """
    Evaluates trained model calibration and performance
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize model evaluator

        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml_config(config_path)
        self.model_params = self.config.get('model_params', {})

        # Create reports directory
        self.reports_dir = Path('reports/model_evaluation')
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_latest_model(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Load the latest trained model

        Returns:
            Tuple of (model, metadata)
        """
        model_dir = Path(self.model_params.get('model_save_path', 'models/artifacts/'))

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} not found")

        # Find latest model file
        model_files = list(model_dir.glob("model_*.pkl"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")

        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

        logger.info(f"Loading model from {latest_model}")

        with open(latest_model, 'rb') as f:
            model_data = pickle.load(f)

        return model_data['model'], model_data['metadata']

    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load test data for evaluation (fallback to training data if no test data)

        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Loading test data")

        test_start = self.model_params.get('test_start_date', '2024-07-01')
        test_end = self.model_params.get('test_end_date', '2024-12-31')

        df = load_features_for_training(test_start, test_end)

        if df.empty:
            # Fallback to training data for evaluation
            logger.warning("No test data found, using training data for evaluation")
            train_start = self.model_params.get('training_start_date', '2020-01-01')
            train_end = self.model_params.get('training_end_date', '2023-12-31')
            df = load_features_for_training(train_start, train_end)

        if df.empty:
            raise ValueError("No data found for evaluation")

        # Extract features and labels
        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols]
        y = df['label_up_10pct']

        # Handle missing values
        X = self._handle_missing_values(X)

        logger.info(f"Loaded {len(X)} evaluation samples with {len(feature_cols)} features")
        return X, y

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature column names"""
        exclude_cols = [
            'symbol', 'date',
            'label_up_10pct', 'label_win_before_loss', 'label_loss_before_win',
            'forward_5d_ret_pct', 'forward_10d_ret_pct', 'forward_10d_max_return_pct'
        ]
        return [col for col in df.columns if col not in exclude_cols]

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        return X.fillna(0).replace([np.inf, -np.inf], 0)

    def analyze_calibration(self, model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Analyze model calibration by probability buckets

        Args:
            model: Trained model
            X: Feature data
            y: True labels

        Returns:
            DataFrame with calibration analysis
        """
        logger.info("Analyzing model calibration")

        # Get predicted probabilities
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Create probability buckets
        bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = ['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']

        # Bucket the predictions
        prob_buckets = pd.cut(y_pred_proba, bins=bins, labels=labels, include_lowest=True)

        # Calculate calibration metrics for each bucket
        calibration_data = []

        for bucket in labels:
            bucket_mask = prob_buckets == bucket
            bucket_size = bucket_mask.sum()

            if bucket_size == 0:
                continue

            bucket_preds = y_pred_proba[bucket_mask]
            bucket_actuals = y[bucket_mask]

            avg_predicted = bucket_preds.mean()
            actual_positive_rate = bucket_actuals.mean()
            calibration_error = abs(avg_predicted - actual_positive_rate)

            calibration_data.append({
                'probability_bucket': bucket,
                'count': bucket_size,
                'avg_predicted_prob': avg_predicted,
                'actual_positive_rate': actual_positive_rate,
                'calibration_error': calibration_error,
                'bucket_range': bucket
            })

        calibration_df = pd.DataFrame(calibration_data)

        # Calculate overall Brier score
        brier_score = brier_score_loss(y, y_pred_proba)
        calibration_df.attrs['brier_score'] = brier_score

        logger.info(f"Calibration analysis complete. Brier score: {brier_score:.4f}")
        return calibration_df

    def create_calibration_report(self, calibration_df: pd.DataFrame) -> str:
        """
        Create calibration report in Markdown format

        Args:
            calibration_df: Calibration analysis DataFrame

        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"calibration_report_{timestamp}.md"

        brier_score = calibration_df.attrs.get('brier_score', 0)

        report = f"""# Model Calibration Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Brier Score:** {brier_score:.4f}

## Overview

This report analyzes the calibration of the trading signal prediction model.
Calibration measures how well predicted probabilities match actual outcomes.

- **Brier Score**: {brier_score:.4f} (lower is better, 0.0 = perfect calibration)
- **Analysis Period**: {self.model_params.get('test_start_date')} to {self.model_params.get('test_end_date')}

## Calibration Table

| Probability Bucket | Count | Avg Predicted | Actual Rate | Calibration Error |
|-------------------|-------|---------------|-------------|-------------------|
"""

        for _, row in calibration_df.iterrows():
            report += f"|{row['probability_bucket']}|{row['count']}|{row['avg_predicted_prob']:.3f}|{row['actual_positive_rate']:.3f}|{row['calibration_error']:.3f}|\n"

        # Add interpretation
        report += """

## Interpretation

### What This Means
- **Predicted Probability**: What the model thinks is the chance of +10% return
- **Actual Rate**: What actually happened in historical data
- **Calibration Error**: How far off the model's probability estimates are

### Good Calibration Indicators
- Low Brier score (< 0.1 is good, < 0.05 is excellent)
- Small calibration errors across buckets
- Predicted probabilities close to actual rates

### Trading Implications
- **0.8-0.9 bucket**: If actual rate is 70-80%, these signals are reliable
- **0.5-0.6 bucket**: If actual rate is ~50%, these are coin-flip trades
- Use higher probability thresholds for more reliable signals

## Recommendations

1. **For Decision Support**: Use 0.8+ probabilities for high-confidence signals
2. **For Risk Management**: Monitor calibration drift over time
3. **For Model Improvement**: Focus on buckets with high calibration error
"""

        # Save report
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Calibration report saved to {report_path}")
        return str(report_path)

    def save_calibration_csv(self, calibration_df: pd.DataFrame) -> str:
        """
        Save calibration data as CSV

        Args:
            calibration_df: Calibration analysis DataFrame

        Returns:
            Path to saved CSV
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.reports_dir / f"calibration_data_{timestamp}.csv"

        calibration_df.to_csv(csv_path, index=False)
        logger.info(f"Calibration CSV saved to {csv_path}")

        return str(csv_path)

    def print_calibration_summary(self, calibration_df: pd.DataFrame):
        """
        Print calibration summary to console

        Args:
            calibration_df: Calibration analysis DataFrame
        """
        print("\n" + "="*70)
        print("🎯 MODEL CALIBRATION ANALYSIS")
        print("="*70)

        brier_score = calibration_df.attrs.get('brier_score', 0)
        print(f"\n📊 Overall Brier Score: {brier_score:.4f}")

        print("\n📋 Calibration by Probability Bucket:")
        print("-" * 70)
        print("Bucket    | Count | Pred Prob | Actual Rate | Error")
        print("-" * 70)

        for _, row in calibration_df.iterrows():
            print("10")

        # Analyze useful buckets
        high_confidence = calibration_df[calibration_df['probability_bucket'].isin(['0.8-0.9', '0.9-1.0'])]

        if not high_confidence.empty:
            avg_high_conf = high_confidence['actual_positive_rate'].mean()
            print(".1f")
            print("   💡 These are your most reliable signals!")

        # Interpretation
        print("\n💡 Interpretation:")
        if brier_score < 0.05:
            print("   ✅ Excellent calibration! Model probabilities are very reliable.")
        elif brier_score < 0.1:
            print("   👍 Good calibration. Model probabilities are reasonably reliable.")
        elif brier_score < 0.2:
            print("   ⚠️ Moderate calibration. Use higher probability thresholds.")
        else:
            print("   ❌ Poor calibration. Model needs improvement.")

    def analyze_prediction_quality(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Analyze prediction quality at different thresholds

        Args:
            model: Trained model
            X: Feature data
            y: True labels

        Returns:
            Dictionary with threshold analysis
        """
        y_pred_proba = model.predict_proba(X)[:, 1]

        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        analysis = {}

        for threshold in thresholds:
            mask = y_pred_proba >= threshold
            if mask.sum() == 0:
                continue

            predicted_positives = mask.sum()
            actual_positives = y[mask].sum()
            hit_rate = actual_positives / predicted_positives if predicted_positives > 0 else 0

            analysis[f'{threshold:.1f}'] = {
                'threshold': threshold,
                'signals_generated': predicted_positives,
                'actual_hits': actual_positives,
                'hit_rate': hit_rate,
                'hit_rate_pct': hit_rate * 100
            }

        return analysis

def main():
    """Main evaluation function"""
    print("🚀 NIFTY TRADING AGENT - MODEL EVALUATION")
    print("=" * 50)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/model_evaluation.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Initialize evaluator
        evaluator = ModelEvaluator()

        # Load model and test data
        model, metadata = evaluator.load_latest_model()
        X_test, y_test = evaluator.load_test_data()

        logger.info(f"Evaluating model {metadata.get('model_version', 'unknown')}")
        logger.info(f"Test period: {evaluator.model_params.get('test_start_date')} to {evaluator.model_params.get('test_end_date')}")

        # Analyze calibration
        calibration_df = evaluator.analyze_calibration(model, X_test, y_test)

        # Print summary
        evaluator.print_calibration_summary(calibration_df)

        # Analyze prediction quality at different thresholds
        threshold_analysis = evaluator.analyze_prediction_quality(model, X_test, y_test)

        print("\n🎯 Prediction Quality by Threshold:")
        print("-" * 60)
        print("Threshold | Signals | Hits | Hit Rate")
        print("-" * 60)

        for threshold, metrics in threshold_analysis.items():
            print("6.1f")

        # Save reports
        csv_path = evaluator.save_calibration_csv(calibration_df)
        md_path = evaluator.create_calibration_report(calibration_df)

        print(f"\n💾 Reports saved:")
        print(f"   CSV: {csv_path}")
        print(f"   Markdown: {md_path}")

        print("\n✅ Model evaluation completed successfully!")
        print("\nNext steps:")
        print("1. Review calibration results")
        print("2. Run backtesting: python -m backtest.backtest_signals_with_model")

        return 0

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}", exc_info=True)
        print(f"❌ Model evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
