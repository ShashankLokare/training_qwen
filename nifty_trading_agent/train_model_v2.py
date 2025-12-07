#!/usr/bin/env python3
"""
Model Training Script v2 for Nifty Trading Agent
Trains v2 ML model with proper imbalance handling and calibration
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    brier_score_loss, classification_report, confusion_matrix,
    precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import load_features_for_training
from utils.io_utils import load_yaml_config
from models.imbalance_utils import ImbalanceHandler

logger = get_logger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using LightGBM")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

class ModelTrainerV2:
    """
    Trains v2 ML model with imbalance handling and calibration
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize v2 model trainer

        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml_config(config_path)
        self.model_params = self.config.get('model_params_v2', self.config.get('model_params', {}))
        self.imbalance_handler = ImbalanceHandler()

        # Create model artifacts directory
        self.model_dir = Path(self.model_params.get('model_save_path', 'models/artifacts_v2/'))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Determine which model to use
        self.model_type = self.model_params.get('model_type', 'xgboost')  # xgboost or lightgbm
        self.primary_label = self.model_params.get('primary_label', 'label_5p_10d')

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load v2 training data from DuckDB

        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Loading v2 training data from DuckDB")

        # Get date ranges for v2
        train_start = self.model_params.get('training_start_date', '2015-01-01')
        train_end = self.model_params.get('training_end_date', '2021-12-31')

        # Load features (should include v2 labels)
        df = load_features_for_training(train_start, train_end)

        if df.empty:
            raise ValueError("No v2 training data found")

        # Check if v2 labels exist
        if self.primary_label not in df.columns:
            raise ValueError(f"Primary label '{self.primary_label}' not found in training data. Run generate_labels_v2.py first.")

        # Extract features and labels
        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols]
        y = df[self.primary_label].astype(int)

        logger.info(f"Loaded {len(X)} training samples with {len(feature_cols)} features")
        logger.info(f"Primary label: {self.primary_label}")

        # Handle missing values
        X = self._handle_missing_values(X)

        return X, y

    def load_validation_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load v2 validation data

        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Loading v2 validation data")

        val_start = self.model_params.get('validation_start_date', '2022-01-01')
        val_end = self.model_params.get('validation_end_date', '2022-12-31')

        df = load_features_for_training(val_start, val_end)

        if df.empty:
            logger.warning("No v2 validation data found")
            return pd.DataFrame(), pd.Series()

        if self.primary_label not in df.columns:
            logger.warning(f"Primary label '{self.primary_label}' not found in validation data")
            return pd.DataFrame(), pd.Series()

        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols]
        y = df[self.primary_label].astype(int)

        X = self._handle_missing_values(X)

        logger.info(f"Loaded {len(X)} validation samples")
        return X, y

    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load v2 test data

        Returns:
            Tuple of (features_df, labels_series)
        """
        logger.info("Loading v2 test data")

        test_start = self.model_params.get('test_start_date', '2023-01-01')
        test_end = self.model_params.get('test_end_date', '2024-12-31')

        df = load_features_for_training(test_start, test_end)

        if df.empty:
            logger.warning("No v2 test data found")
            return pd.DataFrame(), pd.Series()

        if self.primary_label not in df.columns:
            logger.warning(f"Primary label '{self.primary_label}' not found in test data")
            return pd.DataFrame(), pd.Series()

        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols]
        y = df[self.primary_label].astype(int)

        X = self._handle_missing_values(X)

        logger.info(f"Loaded {len(X)} test samples")
        return X, y

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get v2 feature columns (exclude labels and metadata)

        Args:
            df: DataFrame with all columns

        Returns:
            List of feature column names
        """
        exclude_cols = [
            'symbol', 'date',  # Metadata
            # v1 labels
            'label_up_10pct', 'label_win_before_loss', 'label_loss_before_win',
            # v2 labels
            'label_3p_5d', 'label_5p_10d', 'label_outperf_5d',
            'fwd_5d_return', 'fwd_10d_return', 'idx_fwd_5d_return',
            # Forward-looking data
            'forward_5d_ret_pct', 'forward_10d_ret_pct', 'forward_10d_max_return_pct'
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
        # Fill NaN with 0 (could be improved with more sophisticated imputation)
        X_filled = X.fillna(0)

        # Ensure no infinite values
        X_filled = X_filled.replace([np.inf, -np.inf], 0)

        return X_filled

    def train_base_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train the base v2 model (XGBoost or LightGBM)

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        logger.info(f"Training base {self.model_type} model with imbalance handling")

        # Get scale_pos_weight for imbalance handling
        scale_pos_weight = self.imbalance_handler.get_xgboost_scale_pos_weight(y_train)
        logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = self._train_xgboost_model(X_train, y_train, scale_pos_weight)
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            model = self._train_lightgbm_model(X_train, y_train, scale_pos_weight)
        else:
            # Fallback to available model
            if XGBOOST_AVAILABLE:
                logger.warning(f"Requested {self.model_type} not available, using XGBoost")
                self.model_type = 'xgboost'
                model = self._train_xgboost_model(X_train, y_train, scale_pos_weight)
            elif LIGHTGBM_AVAILABLE:
                logger.warning(f"Requested {self.model_type} not available, using LightGBM")
                self.model_type = 'lightgbm'
                model = self._train_lightgbm_model(X_train, y_train, scale_pos_weight)
            else:
                raise ImportError("Neither XGBoost nor LightGBM available")

        logger.info(f"Base {self.model_type} model training completed")
        return model

    def _train_xgboost_model(self, X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float):
        """Train XGBoost model"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 1
        }

        # Override with config params if available
        xgb_params = self.model_params.get('xgboost', {})
        params.update(xgb_params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        return model

    def _train_lightgbm_model(self, X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float):
        """Train LightGBM model"""
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': -1
        }

        # Override with config params if available
        lgb_params = self.model_params.get('lightgbm', {})
        params.update(lgb_params)

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        return model

    def calibrate_model(self, base_model, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, bool]:
        """
        Calibrate the model using Platt scaling if AUC > 0.55

        Args:
            base_model: Trained base model
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Tuple of (calibrated_model, was_calibrated)
        """
        if X_val.empty or y_val.empty:
            logger.warning("No validation data, skipping calibration")
            return base_model, False

        # Evaluate base model on validation set
        val_metrics = self.evaluate_model(base_model, X_val, y_val, "Validation")

        if val_metrics['roc_auc'] < 0.55:
            logger.info(f"Base model AUC {val_metrics['roc_auc']:.3f} < 0.55, skipping calibration")
            return base_model, False

        logger.info(f"Base model AUC {val_metrics['roc_auc']:.3f} >= 0.55, proceeding with calibration")

        # Get predicted probabilities from validation set
        val_probs = base_model.predict_proba(X_val)[:, 1]

        # Train logistic regression on probabilities (Platt scaling)
        platt_model = LogisticRegression(random_state=42)
        platt_model.fit(val_probs.reshape(-1, 1), y_val)

        # Create calibrated model wrapper
        calibrated_model = CalibratedModelWrapper(base_model, platt_model)

        # Verify calibration
        cal_metrics = self.evaluate_model(calibrated_model, X_val, y_val, "Calibrated Validation")

        logger.info(f"Calibration completed. AUC: {val_metrics['roc_auc']:.3f} -> {cal_metrics['roc_auc']:.3f}")
        return calibrated_model, True

    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series,
                      dataset_name: str = "dataset") -> Dict[str, float]:
        """
        Evaluate model performance with imbalance-aware metrics

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

        # Precision@K metrics
        precision_at_5 = self._calculate_precision_at_k(y_pred_proba, y, k=5)
        precision_at_10 = self._calculate_precision_at_k(y_pred_proba, y, k=10)

        metrics['precision@5'] = precision_at_5
        metrics['precision@10'] = precision_at_10

        # F1 Score
        from sklearn.metrics import f1_score
        metrics['f1'] = f1_score(y, y_pred, zero_division=0)

        # PR-AUC
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
        except:
            metrics['pr_auc'] = 0.0

        logger.info(f"{dataset_name} metrics: ROC-AUC={metrics['roc_auc']:.3f}, "
                   f"Precision@5={metrics['precision@5']:.3f}, F1={metrics['f1']:.3f}")

        return metrics

    def _calculate_precision_at_k(self, y_pred_proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
        """Calculate Precision@K"""
        # Get indices of top K predictions
        top_k_indices = np.argsort(y_pred_proba)[::-1][:k]

        # Calculate precision among top K
        if len(top_k_indices) == 0:
            return 0.0

        true_positives = y_true[top_k_indices].sum()
        return true_positives / k

    def save_model(self, model, feature_names: List[str],
                  training_metrics: Dict[str, float],
                  validation_metrics: Dict[str, float] = None,
                  calibrated: bool = False) -> str:
        """
        Save trained v2 model and metadata

        Args:
            model: Trained model
            feature_names: List of feature names used
            training_metrics: Training performance metrics
            validation_metrics: Validation performance metrics
            calibrated: Whether model was calibrated

        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_version = f"v2_{self.model_type}"
        if calibrated:
            model_version += "_calibrated"

        filename = f"model_{model_version}_{timestamp}.pkl"

        model_path = self.model_dir / filename

        # Prepare metadata
        metadata = {
            'model_version': model_version,
            'model_type': self.model_type,
            'calibrated': calibrated,
            'training_timestamp': timestamp,
            'feature_names': feature_names,
            'primary_label': self.primary_label,
            'training_metrics': training_metrics,
            'validation_metrics': validation_metrics,
            'model_params': self.model_params
        }

        # Save model and metadata
        model_data = {
            'model': model,
            'metadata': metadata
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"V2 model saved to {model_path}")
        return str(model_path)

    def print_summary(self, training_metrics: Dict[str, float],
                     validation_metrics: Dict[str, float] = None,
                     test_metrics: Dict[str, float] = None,
                     calibrated: bool = False):
        """
        Print v2 training summary

        Args:
            training_metrics: Training performance
            validation_metrics: Validation performance
            test_metrics: Test performance
            calibrated: Whether model was calibrated
        """
        print("\n" + "="*60)
        print("🤖 MODEL V2 TRAINING SUMMARY")
        print("="*60)

        print(f"\nModel Type: {self.model_type.upper()}")
        print(f"Primary Label: {self.primary_label}")
        print(f"Calibrated: {'Yes' if calibrated else 'No'}")

        print("\n📊 Training Performance:")
        self._print_metrics(training_metrics, "Training")

        if validation_metrics:
            print("\n📊 Validation Performance:")
            self._print_metrics(validation_metrics, "Validation")

        if test_metrics:
            print("\n📊 Test Performance:")
            self._print_metrics(test_metrics, "Test")

    def _print_metrics(self, metrics: Dict[str, float], dataset_name: str):
        """Print metrics for a dataset"""
        print(f"   ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
        print(f"   Precision@5: {metrics.get('precision@5', 0):.3f}")
        print(f"   Precision@10: {metrics.get('precision@10', 0):.3f}")
        print(f"   F1: {metrics.get('f1', 0):.3f}")
        print(f"   PR-AUC: {metrics.get('pr_auc', 0):.3f}")
        print(f"   Precision: {metrics.get('precision', 0):.3f}")
        print(f"   Recall: {metrics.get('recall', 0):.3f}")
        print(f"   Positive Rate: {metrics.get('positive_rate', 0):.1%}")
        print(f"   Sample Size: {metrics.get('sample_size', 0)}")

class CalibratedModelWrapper:
    """
    Wrapper for calibrated models using Platt scaling
    """

    def __init__(self, base_model, platt_model):
        self.base_model = base_model
        self.platt_model = platt_model

    def predict_proba(self, X):
        """Get calibrated probabilities"""
        base_probs = self.base_model.predict_proba(X)[:, 1]
        calibrated_probs = self.platt_model.predict_proba(base_probs.reshape(-1, 1))[:, 1]

        # Return as 2D array [prob_class_0, prob_class_1]
        return np.column_stack([1 - calibrated_probs, calibrated_probs])

    def predict(self, X, threshold=0.5):
        """Get calibrated predictions"""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

def main():
    """Main v2 training function"""
    print("🚀 NIFTY TRADING AGENT - V2 MODEL TRAINING")
    print("=" * 50)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/train_model_v2.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Initialize v2 trainer
        trainer = ModelTrainerV2()

        # Load data
        X_train, y_train = trainer.load_training_data()
        X_val, y_val = trainer.load_validation_data()
        X_test, y_test = trainer.load_test_data()

        if X_train.empty:
            logger.error("No v2 training data available")
            return 1

        # Analyze imbalance
        train_stats = trainer.imbalance_handler.analyze_class_distribution(y_train, "Training")
        trainer.imbalance_handler.print_imbalance_report(train_stats)

        # Train base model
        base_model = trainer.train_base_model(X_train, y_train)

        # Evaluate on training data
        training_metrics = trainer.evaluate_model(base_model, X_train, y_train, "Training")

        # Calibrate model (only if AUC > 0.55 and predicts both classes)
        calibrated_model, was_calibrated = trainer.calibrate_model(base_model, X_val, y_val)

        # Evaluate calibrated model
        validation_metrics = None
        if not X_val.empty:
            validation_metrics = trainer.evaluate_model(calibrated_model, X_val, y_val, "Validation")

        # Evaluate on test data
        test_metrics = None
        if not X_test.empty:
            test_metrics = trainer.evaluate_model(calibrated_model, X_test, y_test, "Test")

        # Save model
        model_path = trainer.save_model(
            calibrated_model,
            X_train.columns.tolist(),
            training_metrics,
            validation_metrics,
            was_calibrated
        )

        # Print summary
        trainer.print_summary(
            training_metrics,
            validation_metrics,
            test_metrics,
            was_calibrated
        )

        print(f"\n💾 Model saved to: {model_path}")
        print("\n✅ V2 model training completed successfully!")
        print("\nNext steps:")
        print("1. Run evaluation: python evaluate_model_v2.py")
        print("2. Run backtesting: python -m backtest.backtest_signals_with_model_v2")

        return 0

    except Exception as e:
        logger.error(f"V2 model training failed: {e}", exc_info=True)
        print(f"❌ V2 model training failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
