#!/usr/bin/env python3
"""
Walk-Forward Validation Engine for Nifty Trading Agent v2
True out-of-sample testing that mimics live trading
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger
from utils.duckdb_tools import load_features_for_training
from regime.regime_detector import RegimeDetector
from models.imbalance_utils import ImbalanceHandler

logger = get_logger(__name__)

class WalkForwardValidator:
    """
    True out-of-sample validation that mimics live trading
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the walk-forward validator

        Args:
            config_path: Path to configuration file
        """
        self.config = pd.io.yaml.load_yaml(config_path) if hasattr(pd.io.yaml, 'load_yaml') else {}
        self.regime_detector = RegimeDetector()
        self.imbalance_handler = ImbalanceHandler()

        # Walk-forward parameters
        self.initial_train_months = 24  # 2 years initial training
        self.validation_months = 3      # 3 months validation
        self.step_months = 1           # Move forward 1 month each step

        logger.info("WalkForwardValidator initialized")

    def run_walk_forward_validation(self,
                                 initial_train_months: int = 24,
                                 validation_months: int = 3,
                                 step_months: int = 1) -> Dict[str, Any]:
        """
        Run comprehensive walk-forward validation

        Args:
            initial_train_months: Initial training window in months
            validation_months: Validation window in months
            step_months: Step size for rolling window

        Returns:
            Dictionary with validation results
        """
        print("🔄 RUNNING WALK-FORWARD VALIDATION")
        print("=" * 40)

        # Load all available data
        all_data = self._load_all_historical_data()
        if all_data.empty:
            return {'error': 'No historical data available'}

        # Create monthly windows for walk-forward
        monthly_windows = self._create_monthly_windows(all_data, initial_train_months, validation_months, step_months)

        validation_results = []

        for i, (train_start, train_end, val_start, val_end) in enumerate(monthly_windows):
            print(f"\n📊 Window {i+1}/{len(monthly_windows)}: Train {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}, Validate {val_start.strftime('%Y-%m')} to {val_end.strftime('%Y-%m')}")

            try:
                # Train model on historical data
                model, feature_names = self._train_model_on_window(train_start, train_end)

                # Validate on future unseen data
                window_result = self._validate_model_on_window(model, feature_names, val_start, val_end)
                validation_results.append(window_result)

            except Exception as e:
                logger.warning(f"Failed validation for window {i+1}: {e}")
                continue

        # Aggregate results
        aggregate_results = self._aggregate_validation_results(validation_results)

        # Generate comprehensive report
        self._generate_walk_forward_report(aggregate_results, validation_results)

        return aggregate_results

    def _load_all_historical_data(self) -> pd.DataFrame:
        """Load all available historical data"""
        try:
            # Load maximum historical range
            start_date = "2010-01-01"  # Go back as far as possible
            end_date = datetime.now().strftime('%Y-%m-%d')

            data = load_features_for_training(start_date, end_date)
            logger.info(f"Loaded {len(data)} historical records from {start_date} to {end_date}")
            return data

        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return pd.DataFrame()

    def _create_monthly_windows(self, data: pd.DataFrame,
                              initial_train_months: int,
                              validation_months: int,
                              step_months: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Create rolling monthly windows for walk-forward validation"""
        if data.empty or 'date' not in data.columns:
            return []

        # Get date range
        min_date = data['date'].min()
        max_date = data['date'].max()

        windows = []
        current_train_end = min_date + pd.DateOffset(months=initial_train_months)

        while current_train_end + pd.DateOffset(months=validation_months) <= max_date:
            train_start = current_train_end - pd.DateOffset(months=initial_train_months)
            train_end = current_train_end
            val_start = current_train_end
            val_end = current_train_end + pd.DateOffset(months=validation_months)

            windows.append((train_start, train_end, val_start, val_end))
            current_train_end += pd.DateOffset(months=step_months)

        logger.info(f"Created {len(windows)} walk-forward windows")
        return windows

    def _train_model_on_window(self, train_start: pd.Timestamp,
                             train_end: pd.Timestamp) -> Tuple[Any, List[str]]:
        """Train model on specific time window"""
        from train_model_v2 import ModelTrainerV2

        try:
            # Create temporary config for this window
            window_config = self.config.copy()
            window_config['model_params_v2'] = self.config.get('model_params_v2', {}).copy()
            window_config['model_params_v2']['training_start_date'] = train_start.strftime('%Y-%m-%d')
            window_config['model_params_v2']['training_end_date'] = train_end.strftime('%Y-%m-%d')

            # Train model
            trainer = ModelTrainerV2()
            trainer.config = window_config

            X_train, y_train = trainer.load_training_data()
            if X_train.empty:
                raise ValueError("No training data for window")

            model = trainer.train_base_model(X_train, y_train)

            return model, X_train.columns.tolist()

        except Exception as e:
            logger.error(f"Model training failed for window {train_start} to {train_end}: {e}")
            raise

    def _validate_model_on_window(self, model, feature_names: List[str],
                                val_start: pd.Timestamp, val_end: pd.Timestamp) -> Dict[str, Any]:
        """Validate model on unseen future data"""
        try:
            # Load validation data
            val_data = load_features_for_training(val_start.strftime('%Y-%m-%d'), val_end.strftime('%Y-%m-%d'))

            if val_data.empty:
                return {'error': 'No validation data'}

            # Ensure we have the required label
            primary_label = self.config.get('model_params_v2', {}).get('primary_label', 'label_5p_10d')
            if primary_label not in val_data.columns:
                return {'error': f'Missing label {primary_label}'}

            # Extract features and labels
            X_val = val_data[feature_names]
            y_val = val_data[primary_label].astype(int)

            # Handle missing values
            X_val = X_val.fillna(0)
            X_val = X_val.replace([np.inf, -np.inf], 0)

            # Get predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Calculate comprehensive metrics
            metrics = self._calculate_window_metrics(y_val, y_pred, y_pred_proba, val_data)

            # Add window metadata
            metrics.update({
                'window_start': val_start,
                'window_end': val_end,
                'sample_size': len(val_data),
                'positive_rate': y_val.mean(),
                'regime': self.regime_detector.get_regime_for_date(val_start + (val_end - val_start) / 2)
            })

            return metrics

        except Exception as e:
            logger.error(f"Validation failed for window {val_start} to {val_end}: {e}")
            return {'error': str(e)}

    def _calculate_window_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                y_pred_proba: np.ndarray, val_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics for validation window"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, roc_auc_score,
            brier_score_loss, f1_score, precision_recall_curve, auc
        )

        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)

        # Precision@K metrics (important for trading)
        metrics['precision@5'] = self._calculate_precision_at_k(y_pred_proba, y_true, 5)
        metrics['precision@10'] = self._calculate_precision_at_k(y_pred_proba, y_true, 10)
        metrics['precision@20'] = self._calculate_precision_at_k(y_pred_proba, y_true, 20)

        # PR-AUC
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
        except:
            metrics['pr_auc'] = 0.0

        # Trading-specific metrics
        trading_metrics = self._calculate_trading_metrics(y_pred_proba, val_data)
        metrics.update(trading_metrics)

        return metrics

    def _calculate_precision_at_k(self, y_pred_proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
        """Calculate Precision@K"""
        if len(y_true) < k:
            k = len(y_true)

        # Get indices of top K predictions
        top_k_indices = np.argsort(y_pred_proba)[::-1][:k]

        if len(top_k_indices) == 0:
            return 0.0

        # Calculate precision among top K
        true_positives = y_true[top_k_indices].sum()
        return true_positives / k

    def _calculate_trading_metrics(self, y_pred_proba: np.ndarray, val_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate trading-specific performance metrics"""
        metrics = {}

        # Simulate trading with different conviction thresholds
        conviction_thresholds = [0.5, 0.6, 0.7, 0.8]

        for threshold in conviction_thresholds:
            signals = y_pred_proba >= threshold
            if signals.sum() > 0:
                # Calculate hit rate for signals above threshold
                hits = val_data.loc[signals, 'label_5p_10d'].mean()
                metrics[f'hit_rate_{int(threshold*100)}'] = hits
                metrics[f'signal_rate_{int(threshold*100)}'] = signals.mean()

        # Information coefficient (correlation between predictions and actual outcomes)
        actual_outcomes = val_data['label_5p_10d'].values
        ic = np.corrcoef(y_pred_proba, actual_outcomes)[0, 1]
        metrics['information_coefficient'] = ic

        # Rank correlation (Spearman)
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(y_pred_proba, actual_outcomes)
        metrics['spearman_correlation'] = spearman_corr

        return metrics

    def _aggregate_validation_results(self, validation_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across all validation windows"""
        if not validation_results:
            return {'error': 'No validation results to aggregate'}

        # Filter out error results
        valid_results = [r for r in validation_results if 'error' not in r]

        if not valid_results:
            return {'error': 'All validation windows failed'}

        # Aggregate metrics
        aggregated = {}
        metric_keys = [k for k in valid_results[0].keys() if isinstance(valid_results[0][k], (int, float)) and k != 'sample_size']

        for metric in metric_keys:
            values = [r[metric] for r in valid_results if metric in r]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)

        # Overall statistics
        aggregated.update({
            'total_windows': len(validation_results),
            'successful_windows': len(valid_results),
            'success_rate': len(valid_results) / len(validation_results),
            'average_sample_size': np.mean([r.get('sample_size', 0) for r in valid_results]),
            'consistency_score': self._calculate_consistency_score(valid_results)
        })

        return aggregated

    def _calculate_consistency_score(self, results: List[Dict]) -> float:
        """Calculate consistency score across validation windows"""
        if len(results) < 2:
            return 1.0

        # Calculate coefficient of variation for key metrics
        key_metrics = ['roc_auc', 'precision@10', 'f1']
        cv_scores = []

        for metric in key_metrics:
            values = [r[metric] for r in results if metric in r]
            if len(values) > 1:
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                cv_scores.append(cv)

        # Lower CV = higher consistency (invert and scale to 0-1)
        avg_cv = np.mean(cv_scores) if cv_scores else 0
        consistency_score = 1 / (1 + avg_cv)  # Ranges from 0 to 1

        return consistency_score

    def _generate_walk_forward_report(self, aggregate_results: Dict,
                                    individual_results: List[Dict]):
        """Generate comprehensive walk-forward validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path("reports/walk_forward_validation/") / f"wf_validation_report_{timestamp}.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WALK-FORWARD VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total validation windows: {aggregate_results.get('total_windows', 0)}\n")
            f.write(f"Successful validations: {aggregate_results.get('successful_windows', 0)}\n")
            f.write(f"Success rate: {aggregate_results.get('success_rate', 0):.1%}\n")
            f.write(f"Consistency score: {aggregate_results.get('consistency_score', 0):.3f}\n\n")

            # Key metrics
            f.write("AGGREGATED PERFORMANCE METRICS\n")
            f.write("-" * 35 + "\n")

            key_metrics = [
                ('ROC-AUC', 'roc_auc'),
                ('Precision@10', 'precision@10'),
                ('F1 Score', 'f1'),
                ('Information Coefficient', 'information_coefficient')
            ]

            for display_name, metric_key in key_metrics:
                mean_key = f'{metric_key}_mean'
                std_key = f'{metric_key}_std'

                if mean_key in aggregate_results:
                    mean_val = aggregate_results[mean_key]
                    std_val = aggregate_results.get(std_key, 0)
                    f.write(f"{display_name}: {mean_val:.3f} ± {std_val:.3f}\n")

            f.write("\nTRADING PERFORMANCE\n")
            f.write("-" * 20 + "\n")

            # Hit rates by conviction threshold
            for threshold in [50, 60, 70, 80]:
                hit_rate_key = f'hit_rate_{threshold}_mean'
                signal_rate_key = f'signal_rate_{threshold}_mean'

                if hit_rate_key in aggregate_results:
                    hit_rate = aggregate_results[hit_rate_key]
                    signal_rate = aggregate_results.get(signal_rate_key, 0)
                    f.write(f"Threshold {threshold}%: {hit_rate:.1%} hit rate ({signal_rate:.1%} signal rate)\n")

            f.write("\nINDIVIDUAL WINDOW RESULTS\n")
            f.write("-" * 27 + "\n")
            f.write("<10")
            f.write("-" * 80 + "\n")

            for i, result in enumerate(individual_results[:20]):  # Show first 20
                if 'error' not in result:
                    window_start = result.get('window_start', 'N/A')
                    roc_auc = result.get('roc_auc', 0)
                    precision_10 = result.get('precision@10', 0)
                    regime = result.get('regime', 'unknown')

                    f.write("<10")
                else:
                    f.write(f"Window {i+1}: ERROR - {result['error']}\n")

            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")

            consistency = aggregate_results.get('consistency_score', 0)
            avg_roc_auc = aggregate_results.get('roc_auc_mean', 0)
            avg_precision_10 = aggregate_results.get('precision@10_mean', 0)

            if consistency > 0.8 and avg_roc_auc > 0.6 and avg_precision_10 > 0.15:
                f.write("✅ EXCELLENT: System shows strong out-of-sample performance\n")
                f.write("   Ready for live trading with confidence\n")
            elif consistency > 0.6 and avg_roc_auc > 0.55:
                f.write("⚠️  GOOD: System shows reasonable performance\n")
                f.write("   Consider additional validation before full deployment\n")
            else:
                f.write("❌ CONCERNING: System shows poor out-of-sample performance\n")
                f.write("   Significant improvements needed before live trading\n")

        print(f"📄 Walk-forward validation report saved to: {report_path}")

        # Also save as JSON for programmatic access
        json_path = report_path.with_suffix('.json')
        import json

        # Convert timestamps and numpy types for JSON serialization
        def serialize_for_json(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(json_path, 'w') as f:
            json.dump({
                'aggregate_results': aggregate_results,
                'individual_results': individual_results,
                'generated_at': datetime.now().isoformat()
            }, f, indent=2, default=serialize_for_json)

def main():
    """Main walk-forward validation function"""
    print("🔄 NIFTY TRADING AGENT - WALK-FORWARD VALIDATION")
    print("=" * 55)

    try:
        # Initialize validator
        validator = WalkForwardValidator()

        # Run walk-forward validation
        results = validator.run_walk_forward_validation(
            initial_train_months=24,  # 2 years training
            validation_months=3,     # 3 months validation
            step_months=1           # Move forward 1 month each step
        )

        if 'error' in results:
            print(f"❌ Walk-forward validation failed: {results['error']}")
            return 1

        # Print summary
        print("\n🎯 WALK-FORWARD VALIDATION SUMMARY")
        print("=" * 40)

        print("Key Metrics:")
        print(f"  Total Windows: {results.get('total_windows', 0)}")
        print(f"  Success Rate: {results.get('success_rate', 0):.1%}")
        print(f"  Consistency Score: {results.get('consistency_score', 0):.3f}")
        print(f"  Average ROC-AUC: {results.get('roc_auc_mean', 0):.3f}")
        print(f"  Average Precision@10: {results.get('precision@10_mean', 0):.3f}")

        consistency = results.get('consistency_score', 0)
        avg_roc_auc = results.get('roc_auc_mean', 0)

        if consistency > 0.8 and avg_roc_auc > 0.6:
            print("\n✅ EXCELLENT: Walk-forward validation shows strong out-of-sample performance!")
            print("   System is ready for live trading.")
        elif consistency > 0.6 and avg_roc_auc > 0.55:
            print("\n⚠️  GOOD: Reasonable out-of-sample performance.")
            print("   Consider additional testing before full deployment.")
        else:
            print("\n❌ CONCERNING: Poor out-of-sample performance detected.")
            print("   Significant improvements needed before live trading.")

        print("\n✅ Walk-forward validation completed!")
        return 0

    except Exception as e:
        print(f"❌ Walk-forward validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
