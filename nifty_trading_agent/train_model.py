#!/usr/bin/env python3
"""
Model Training Script for Nifty Trading Agent
Trains ML model to predict probability of +10% return within 10 days
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    brier_score_loss, classification_report
)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import load_features_for_training
from utils.io_utils import load_yaml_config

logger = get_logger(__name__)

class ModelTrainer:
    """
    Trains and calibrates ML model for trading signal prediction
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize model trainer

        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml_config(config_path)
        self.model_params = self.config.get('model_params', {})
        self.rf_params = self.model_params.get('random_forest', {})
        self.calibration_method = self.model_params.get('calibration_method', 'isotonic')

        # Create model artifacts directory
        self.model_dir = Path(self.model_params.get('model_save_path', 'models/artifacts/'))
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training data from DuckDB

        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Loading training data from DuckDB")

        # Get date ranges
        train_start = self.model_params.get('training_start_date', '2020-01-01')
        train_end = self.model_params.get('training_end_date', '2023-12-31')

        # Load features
        df = load_features_for_training(train_start, train_end)

        if df.empty:
            raise ValueError("No training data found")

        # Extract features and labels
        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols]
        y = df['label_up_10pct']

        logger.info(f"Loaded {len(X)} training samples with {len(feature_cols)} features")

        # Handle missing values
        X = self._handle_missing_values(X)

        return X, y

    def load_validation_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load validation data for calibration

        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Loading validation data from DuckDB")

        val_start = self.model_params.get('validation_start_date', '2024-01-01')
        val_end = self.model_params.get('validation_end_date', '2024-06-30')

        df = load_features_for_training(val_start, val_end)

        if df.empty:
            logger.warning("No validation data found, skipping calibration")
            return pd.DataFrame(), pd.Series()

        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols]
        y = df['label_up_10pct']

        X = self._handle_missing_values(X)

        logger.info(f"Loaded {len(X)} validation samples")
        return X, y

    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load test data for evaluation

        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Loading test data from DuckDB")

        test_start = self.model_params.get('test_start_date', '2024-07-01')
        test_end = self.model_params.get('test_end_date', '2024-12-31')

        df = load_features_for_training(test_start, test_end)

        if df.empty:
            logger.warning("No test data found")
            return pd.DataFrame(), pd.Series()

        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols]
        y = df['label_up_10pct']

        X = self._handle_missing_values(X)

        logger.info(f"Loaded {len(X)} test samples")
        return X, y

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (exclude labels and metadata)

        Args:
            df: DataFrame with all columns

        Returns:
            List of feature column names
        """
        exclude_cols = [
            'symbol', 'date',  # Metadata
            'label_up_10pct', 'label_win_before_loss', 'label_loss_before_win',  # Labels
            'forward_5d_ret_pct', 'forward_10d_ret_pct', 'forward_10d_max_return_pct'  # Forward-looking
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in features

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with missing values handled
        """
        # Fill NaN with 0 for now (could be improved with more sophisticated imputation)
        X_filled = X.fillna(0)

        # Ensure no infinite values
        X_filled = X_filled.replace([np.inf, -np.inf], 0)

        return X_filled

    def train_base_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Train the base Random Forest model

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained RandomForestClassifier
        """
        logger.info("Training base Random Forest model")

        model = RandomForestClassifier(**self.rf_params)
        model.fit(X_train, y_train)

        logger.info("Base model training completed")
        return model

    def calibrate_model(self, base_model: RandomForestClassifier,
                       X_val: pd.DataFrame, y_val: pd.Series) -> CalibratedClassifierCV:
        """
        Calibrate the model for probability predictions

        Args:
            base_model: Trained base model
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Calibrated model
        """
        if X_val.empty or y_val.empty:
            logger.warning("No validation data, skipping calibration")
            return base_model

        logger.info(f"Calibrating model using {self.calibration_method} method")

        calibrated_model = CalibratedClassifierCV(
            base_model,
            method=self.calibration_method,
            cv='prefit'  # Model is already fitted
        )

        calibrated_model.fit(X_val, y_val)

        logger.info("Model calibration completed")
        return calibrated_model

    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series,
                      dataset_name: str = "dataset") -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            model: Trained model
            X: Feature data
            y: True labels
            dataset_name: Name for logging

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name}")

        # Get predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'brier_score': brier_score_loss(y, y_pred_proba),
            'positive_rate': y.mean(),
            'sample_size': len(y)
        }

        logger.info(f"{dataset_name} metrics: Accuracy={metrics['accuracy']:.3f}, "
                   f"ROC-AUC={metrics['roc_auc']:.3f}, Brier={metrics['brier_score']:.3f}")

        return metrics

    def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance from the model

        Args:
            model: Trained model
            feature_names: List of feature names

        Returns:
            Dictionary of feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            # For calibrated models, get importance from base estimator
            importance = model.estimators_[0].feature_importances_
        else:
            logger.warning("Could not extract feature importance")
            return {}

        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importance))

        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return importance_dict

    def save_model(self, model, feature_names: List[str],
                  training_metrics: Dict[str, float],
                  validation_metrics: Dict[str, float] = None) -> str:
        """
        Save trained model and metadata

        Args:
            model: Trained model
            feature_names: List of feature names used
            training_metrics: Training performance metrics
            validation_metrics: Validation performance metrics

        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_version = self.model_params.get('model_version', 'v1.0')
        filename = f"model_{model_version}_{timestamp}.pkl"

        model_path = self.model_dir / filename

        # Prepare metadata
        metadata = {
            'model_version': model_version,
            'training_timestamp': timestamp,
            'feature_names': feature_names,
            'training_metrics': training_metrics,
            'validation_metrics': validation_metrics,
            'model_params': self.model_params,
            'calibration_method': self.calibration_method
        }

        # Save model and metadata
        model_data = {
            'model': model,
            'metadata': metadata
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def print_summary(self, training_metrics: Dict[str, float],
                     validation_metrics: Dict[str, float] = None,
                     test_metrics: Dict[str, float] = None,
                     feature_importance: Dict[str, float] = None):
        """
        Print training summary

        Args:
            training_metrics: Training performance
            validation_metrics: Validation performance
            test_metrics: Test performance
            feature_importance: Feature importance dictionary
        """
        print("\n" + "="*60)
        print("🎯 MODEL TRAINING SUMMARY")
        print("="*60)

        print("\n📊 Training Performance:")
        self._print_metrics(training_metrics, "Training")

        if validation_metrics:
            print("\n📊 Validation Performance:")
            self._print_metrics(validation_metrics, "Validation")

        if test_metrics:
            print("\n📊 Test Performance:")
            self._print_metrics(test_metrics, "Test")

        if feature_importance:
            print("\n🔍 Top 10 Important Features:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                print(".3f")

    def _print_metrics(self, metrics: Dict[str, float], dataset_name: str):
        """Print metrics for a dataset"""
        print(f"   Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"   Precision: {metrics.get('precision', 0):.3f}")
        print(f"   Recall: {metrics.get('recall', 0):.3f}")
        print(f"   ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
        print(f"   Brier Score: {metrics.get('brier_score', 0):.3f}")
        print(f"   Positive Rate: {metrics.get('positive_rate', 0):.1%}")
        print(f"   Sample Size: {metrics.get('sample_size', 0)}")

def main():
    """Main training function"""
    print("🚀 NIFTY TRADING AGENT - MODEL TRAINING")
    print("=" * 50)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/model_training.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Initialize trainer
        trainer = ModelTrainer()

        # Load data
        X_train, y_train = trainer.load_training_data()
        X_val, y_val = trainer.load_validation_data()
        X_test, y_test = trainer.load_test_data()

        if X_train.empty:
            logger.error("No training data available")
            return 1

        # Train base model
        base_model = trainer.train_base_model(X_train, y_train)

        # Evaluate on training data
        training_metrics = trainer.evaluate_model(base_model, X_train, y_train, "Training")

        # Calibrate model
        calibrated_model = trainer.calibrate_model(base_model, X_val, y_val)

        # Evaluate calibrated model
        validation_metrics = None
        if not X_val.empty:
            validation_metrics = trainer.evaluate_model(calibrated_model, X_val, y_val, "Validation")

        # Evaluate on test data
        test_metrics = None
        if not X_test.empty:
            test_metrics = trainer.evaluate_model(calibrated_model, X_test, y_test, "Test")

        # Get feature importance
        feature_importance = trainer.get_feature_importance(calibrated_model, X_train.columns.tolist())

        # Save model
        model_path = trainer.save_model(
            calibrated_model,
            X_train.columns.tolist(),
            training_metrics,
            validation_metrics
        )

        # Print summary
        trainer.print_summary(
            training_metrics,
            validation_metrics,
            test_metrics,
            feature_importance
        )

        print(f"\n💾 Model saved to: {model_path}")
        print("\n✅ Model training completed successfully!")
        print("\nNext steps:")
        print("1. Run evaluation: python evaluate_model.py")
        print("2. Run backtesting: python -m backtest.backtest_signals_with_model")

        return 0

    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        print(f"❌ Model training failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
