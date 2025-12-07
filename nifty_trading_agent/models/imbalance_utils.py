#!/usr/bin/env python3
"""
Imbalance Utilities for Nifty Trading Agent v2
Handles class imbalance in ML training with proper weighting calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImbalanceHandler:
    """
    Handles class imbalance for binary classification tasks
    """

    def __init__(self):
        """Initialize imbalance handler"""
        pass

    def analyze_class_distribution(self, y: pd.Series, dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Analyze class distribution and imbalance metrics

        Args:
            y: Target labels (0/1)
            dataset_name: Name for logging

        Returns:
            Dictionary with imbalance statistics
        """
        total_samples = len(y)
        pos_count = y.sum()
        neg_count = total_samples - pos_count

        pos_rate = pos_count / total_samples
        neg_rate = neg_count / total_samples

        # Imbalance ratio
        imbalance_ratio = neg_count / pos_count if pos_count > 0 else float('inf')

        # Calculate scale_pos_weight for XGBoost/LightGBM
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        # Additional metrics
        minority_class = "positive" if pos_count < neg_count else "negative"
        minority_count = min(pos_count, neg_count)
        majority_count = max(pos_count, neg_count)

        stats = {
            'total_samples': total_samples,
            'positive_count': int(pos_count),
            'negative_count': int(neg_count),
            'positive_rate': pos_rate,
            'negative_rate': neg_rate,
            'imbalance_ratio': imbalance_ratio,
            'scale_pos_weight': scale_pos_weight,
            'minority_class': minority_class,
            'minority_count': minority_count,
            'majority_count': majority_count,
            'is_balanced': 0.4 <= pos_rate <= 0.6  # Roughly balanced if 40-60%
        }

        logger.info(f"{dataset_name} class distribution:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Positive: {pos_count} ({pos_rate:.1%})")
        logger.info(f"  Negative: {neg_count} ({neg_rate:.1%})")
        logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        logger.info(f"  Scale pos weight: {scale_pos_weight:.2f}")

        return stats

    def get_xgboost_scale_pos_weight(self, y: pd.Series) -> float:
        """
        Calculate scale_pos_weight for XGBoost

        Args:
            y: Target labels

        Returns:
            Scale positive weight value
        """
        pos_count = y.sum()
        neg_count = len(y) - pos_count

        if pos_count == 0:
            logger.warning("No positive samples found, using scale_pos_weight=1.0")
            return 1.0

        scale_pos_weight = neg_count / pos_count
        return scale_pos_weight

    def get_lightgbm_scale_pos_weight(self, y: pd.Series) -> float:
        """
        Calculate scale_pos_weight for LightGBM (same as XGBoost)

        Args:
            y: Target labels

        Returns:
            Scale positive weight value
        """
        return self.get_xgboost_scale_pos_weight(y)

    def create_class_weight_dict(self, y: pd.Series) -> Dict[int, float]:
        """
        Create class weight dictionary for sklearn models

        Args:
            y: Target labels

        Returns:
            Dictionary with class weights
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)

        class_weight_dict = dict(zip(classes, weights))

        logger.info(f"Class weights: {class_weight_dict}")
        return class_weight_dict

    def print_imbalance_report(self, train_stats: Dict[str, Any],
                              val_stats: Optional[Dict[str, Any]] = None,
                              test_stats: Optional[Dict[str, Any]] = None):
        """
        Print comprehensive imbalance report

        Args:
            train_stats: Training set statistics
            val_stats: Validation set statistics
            test_stats: Test set statistics
        """
        print("\n⚖️  CLASS IMBALANCE ANALYSIS")
        print("=" * 40)

        # Training set
        print("\n📊 Training Set:")
        self._print_dataset_stats(train_stats, "Train")

        if val_stats:
            print("\n📊 Validation Set:")
            self._print_dataset_stats(val_stats, "Validation")

        if test_stats:
            print("\n📊 Test Set:")
            self._print_dataset_stats(test_stats, "Test")

        # Recommendations
        print("\n💡 Recommendations:")
        scale_pos_weight = train_stats['scale_pos_weight']

        if scale_pos_weight > 10:
            print("   • High imbalance detected (ratio > 10:1)")
            print("   • Use scale_pos_weight in XGBoost/LightGBM")
            print("   • Consider data augmentation for positive class")
            print("   • Use focal loss or weighted loss functions")
        elif scale_pos_weight > 5:
            print("   • Moderate imbalance detected (ratio 5-10:1)")
            print("   • Use scale_pos_weight in XGBoost/LightGBM")
            print("   • Monitor for overfitting to majority class")
        elif scale_pos_weight > 2:
            print("   • Mild imbalance detected (ratio 2-5:1)")
            print("   • Use scale_pos_weight or class weights")
            print("   • Should not significantly impact performance")
        else:
            print("   • Relatively balanced classes (ratio < 2:1)")
            print("   • No special handling required")

        # Target range check
        pos_rate = train_stats['positive_rate']
        if 0.05 <= pos_rate <= 0.20:
            print("   • ✅ Positive rate in target range (5-20%)")
        else:
            print(f"   • ⚠️  Positive rate {pos_rate:.1%} outside target range (5-20%)")
            if pos_rate < 0.05:
                print("      → Labels may be too rare, consider lowering thresholds")
            else:
                print("      → Labels may be too common, consider raising thresholds")

    def _print_dataset_stats(self, stats: Dict[str, Any], name: str):
        """Print statistics for a dataset"""
        print(f"   Samples: {stats['total_samples']}")
        print(f"   Positive: {stats['positive_count']} ({stats['positive_rate']:.1%})")
        print(f"   Negative: {stats['negative_count']} ({stats['negative_rate']:.1%})")
        print(f"   Imbalance ratio: {stats['imbalance_ratio']:.2f}")
        print(f"   Scale pos weight: {stats['scale_pos_weight']:.2f}")

        if stats['is_balanced']:
            print("   Balance: Relatively balanced")
        else:
            minority = stats['minority_class']
            print(f"   Balance: Imbalanced ({minority} minority class)")

    def create_sample_weights(self, y: pd.Series) -> np.ndarray:
        """
        Create sample weights for imbalanced learning

        Args:
            y: Target labels

        Returns:
            Array of sample weights
        """
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        total = len(y)

        # Weight inversely proportional to class frequency
        pos_weight = total / (2 * pos_count) if pos_count > 0 else 1.0
        neg_weight = total / (2 * neg_count) if neg_count > 0 else 1.0

        weights = np.where(y == 1, pos_weight, neg_weight)

        logger.info(f"Sample weights - Positive: {pos_weight:.2f}, Negative: {neg_weight:.2f}")
        return weights

    def get_imbalance_aware_metrics(self) -> Dict[str, str]:
        """
        Get recommended evaluation metrics for imbalanced datasets

        Returns:
            Dictionary with metric recommendations
        """
        return {
            'primary_metrics': ['ROC-AUC', 'Precision@5', 'Precision@10', 'F1'],
            'secondary_metrics': ['Accuracy', 'Recall', 'PR-AUC'],
            'avoid_metrics': ['Accuracy (misleading on imbalanced data)'],
            'notes': [
                'ROC-AUC is better than accuracy for imbalanced data',
                'Precision@K metrics show top-K performance',
                'F1 balances precision and recall',
                'Use confusion matrix to understand prediction patterns'
            ]
        }

def analyze_v2_label_imbalance(labels_df: pd.DataFrame,
                               primary_label: str = 'label_5p_10d') -> Dict[str, Any]:
    """
    Analyze imbalance for v2 labels

    Args:
        labels_df: DataFrame with v2 labels
        primary_label: Primary target label column

    Returns:
        Dictionary with imbalance analysis
    """
    handler = ImbalanceHandler()

    if primary_label not in labels_df.columns:
        raise ValueError(f"Primary label '{primary_label}' not found in labels DataFrame")

    y = labels_df[primary_label].astype(int)

    # Overall analysis
    overall_stats = handler.analyze_class_distribution(y, "Overall v2 Labels")

    # Yearly analysis
    yearly_stats = {}
    labels_df['year'] = labels_df['date'].dt.year

    for year, group in labels_df.groupby('year'):
        y_year = group[primary_label].astype(int)
        if len(y_year) > 0:
            yearly_stats[year] = handler.analyze_class_distribution(y_year, f"Year {year}")

    # Symbol analysis (summary)
    symbol_stats = {}
    for symbol, group in labels_df.groupby('symbol'):
        y_symbol = group[primary_label].astype(int)
        if len(y_symbol) > 10:  # Only symbols with sufficient data
            symbol_stats[symbol] = handler.analyze_class_distribution(y_symbol, f"Symbol {symbol}")

    # Summary by symbol
    symbol_pos_rates = {sym: stats['positive_rate'] for sym, stats in symbol_stats.items()}

    analysis = {
        'overall': overall_stats,
        'yearly': yearly_stats,
        'symbol_summary': {
            'total_symbols': len(symbol_stats),
            'avg_positive_rate': np.mean(list(symbol_pos_rates.values())),
            'std_positive_rate': np.std(list(symbol_pos_rates.values())),
            'min_positive_rate': min(symbol_pos_rates.values()),
            'max_positive_rate': max(symbol_pos_rates.values())
        },
        'recommendations': {
            'scale_pos_weight': overall_stats['scale_pos_weight'],
            'use_class_weighting': overall_stats['scale_pos_weight'] > 2,
            'target_positive_rate': 0.05 <= overall_stats['positive_rate'] <= 0.20
        }
    }

    return analysis

def main():
    """Demo function for imbalance analysis"""
    print("⚖️  IMBALANCE UTILS - DEMO")
    print("=" * 30)

    # Create sample imbalanced data
    np.random.seed(42)
    n_samples = 10000
    pos_rate = 0.08  # 8% positive rate

    y = np.random.choice([0, 1], size=n_samples, p=[1-pos_rate, pos_rate])

    handler = ImbalanceHandler()
    stats = handler.analyze_class_distribution(pd.Series(y), "Demo Dataset")

    handler.print_imbalance_report(stats)

    print(f"\nSample weights shape: {handler.create_sample_weights(pd.Series(y)).shape}")
    print(f"Class weight dict: {handler.get_class_weight_dict(pd.Series(y))}")

if __name__ == "__main__":
    main()
