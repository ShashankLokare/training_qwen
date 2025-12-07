#!/usr/bin/env python3
"""
Advanced ML Model Training Script for Nifty Trading Agent
Uses XGBoost/LightGBM with hyperparameter tuning and advanced features
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import load_features_for_training
from utils.io_utils import load_yaml_config

logger = get_logger(__name__)

class AdvancedModelTrainer:
    """
    Advanced ML model training with XGBoost/LightGBM and hyperparameter tuning
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize advanced model trainer

        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml_config(config_path)
        self.model_params = self.config.get('model_params', {})
        self.model_name = "xgboost"  # Options: xgboost, lightgbm

        # Training parameters
        self.target_column = 'label_up_10pct_10d'  # Primary target
        self.prediction_horizon = 10  # Days

        # Model hyperparameters
        self.hyperparams = self._get_model_hyperparams()

        # Feature columns to exclude
        self.exclude_columns = [
            'symbol', 'date', 'label_up_10pct_10d', 'label_up_10pct_20d',
            'label_win_before_loss_10d', 'label_loss_before_win_10d',
            'label_win_before_loss_20d', 'label_loss_before_win_20d',
            'label_low_vol_up_10pct', 'label_high_vol_up_10pct',
            'forward_10d_max_return_pct', 'forward_10d_min_return_pct',
            'forward_10d_volatility', 'forward_20d_max_return_pct',
            'forward_20d_min_return_pct', 'forward_20d_volatility'
        ]

    def _get_model_hyperparams(self) -> Dict[str, Any]:
        """Get model hyperparameters based on model type"""
        if self.model_name == "xgboost":
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 1.0,  # Will be updated based on class balance
                'random_state': 42,
                'verbosity': 1
            }
        elif self.model_name == "lightgbm":
            return {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'scale_pos_weight': 1.0,  # Will be updated based on class balance
                'random_state': 42
            }
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training data with advanced features

        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Loading training data with advanced features...")

        # Load from extended date range for real data
        start_date = "2018-01-01"
        end_date = "2023-12-31"

        df = load_features_for_training(start_date, end_date)

        if df.empty:
            raise ValueError("No training data found")

        logger.info(f"Loaded {len(df)} training samples")

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in self.exclude_columns]
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")

        # Prepare features and labels
        X = df[feature_cols]
        y = df[self.target_column]

        # Handle missing values
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        # Calculate class weights for imbalanced data
        pos_weight = len(y) / (2 * y.sum()) if y.sum() > 0 else 1.0
        self.hyperparams['scale_pos_weight'] = min(pos_weight, 10.0)  # Cap at 10

        logger.info(f"Positive class weight: {self.hyperparams['scale_pos_weight']:.2f}")
        logger.info(f"Positive samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

        return X, y

    def perform_time_series_split(self, X: pd.DataFrame, y: pd.Series,
                                n_splits: int = 5) -> List[Tuple]:
        """
        Perform time series cross-validation splits

        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV splits

        Returns:
            List of (train_idx, val_idx) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        splits = []
        for train_idx, val_idx in tscv.split(X):
            splits.append((train_idx, val_idx))

        return splits

    def train_model(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Train the advanced ML model

        Returns:
            Tuple of (trained_model, training_metadata)
        """
        logger.info(f"Training {self.model_name.upper()} model...")

        # Load data
        X, y = self.load_training_data()

        # Time series cross-validation
        cv_splits = self.perform_time_series_split(X, y, n_splits=3)

        # Train model
        if self.model_name == "xgboost":
            model = xgb.XGBClassifier(**self.hyperparams)
        elif self.model_name == "lightgbm":
            model = lgb.LGBMClassifier(**self.hyperparams)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        # Train with early stopping using last split
        train_idx, val_idx = cv_splits[-1]  # Use last split for final training
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")

        if self.model_name == "xgboost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        else:  # LightGBM
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )

        # Apply probability calibration
        logger.info("Applying probability calibration...")
        calibrated_model = CalibratedClassifierCV(
            model, method='isotonic', cv='prefit'
        )
        calibrated_model.fit(X_val, y_val)

        # Evaluate on validation set
        y_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]
        brier_score = brier_score_loss(y_val, y_pred_proba)

        logger.info(f"Validation Brier Score: {brier_score:.4f}")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"Top 10 features: {[f'{k}({v:.3f})' for k,v in top_features]}")

        # Training metadata
        metadata = {
            'model_name': self.model_name,
            'model_version': f"{self.model_name}_v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'training_date': datetime.now().isoformat(),
            'hyperparameters': self.hyperparams,
            'feature_columns': list(X.columns),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'positive_class_weight': self.hyperparams['scale_pos_weight'],
            'brier_score': brier_score,
            'target_column': self.target_column,
            'prediction_horizon': self.prediction_horizon
        }

        return calibrated_model, metadata

    def save_model(self, model: Any, metadata: Dict[str, Any]) -> str:
        """
        Save trained model and metadata

        Args:
            model: Trained model
            metadata: Training metadata

        Returns:
            Path to saved model
        """
        model_dir = Path('models/artifacts')
        model_dir.mkdir(parents=True, exist_ok=True)

        model_version = metadata['model_version']
        model_path = model_dir / f"{model_version}.pkl"

        model_data = {
            'model': model,
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {model_path}")
        return str(model_path)

def main():
    """Main training function"""
    print("🚀 NIFTY TRADING AGENT - ADVANCED ML MODEL TRAINING")
    print("=" * 60)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/advanced_model_training.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Initialize trainer
        trainer = AdvancedModelTrainer()

        # Train model
        logger.info(f"Starting {trainer.model_name.upper()} model training...")
        model, metadata = trainer.train_model()

        # Save model
        model_path = trainer.save_model(model, metadata)

        # Print results
        print("\n🎯 MODEL TRAINING COMPLETED")
        print("=" * 50)
        print(f"Model: {metadata['model_name'].upper()}")
        print(f"Version: {metadata['model_version']}")
        print(f"Target: {metadata['target_column']}")
        print(f"Training Samples: {metadata['training_samples']:,}")
        print(f"Validation Samples: {metadata['validation_samples']:,}")
        print(f"Brier Score: {metadata['brier_score']:.4f}")
        print(f"Positive Class Weight: {metadata['positive_class_weight']:.2f}")
        print(f"Model Saved: {model_path}")

        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(metadata['feature_columns'], model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\n🔍 Top 10 Features:")
            for i, (feature, importance) in enumerate(top_features, 1):
                print("10")

        print("\n✅ Advanced ML model training completed successfully!")
        print("\nNext steps:")
        print("1. Run python evaluate_model.py (advanced calibration analysis)")
        print("2. Run python -m backtest.backtest_signals_with_model (extended backtest)")
        print("3. Run python interactive_main.py (real conviction scores)")

        return 0

    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        print(f"❌ Model training failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
