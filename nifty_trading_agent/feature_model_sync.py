#!/usr/bin/env python3
"""
FEATURE-MODEL SYNCHRONIZATION VERIFICATION
Ensures perfect alignment between feature generation and ML model inputs

Critical verification for production ML trading systems
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import load_ohlcv
from features_engineering_advanced import AdvancedFeatureEngineer

logger = get_logger(__name__)

class FeatureModelSynchronizer:
    """
    Ensures perfect synchronization between feature generation and model inputs
    Critical for production ML trading systems
    """

    def __init__(self):
        """Initialize synchronizer"""
        self.model = None
        self.model_features = []
        self.feature_generator = AdvancedFeatureEngineer()
        self.sync_report = {}

    def load_trained_model(self):
        """Load the latest trained model and extract feature information"""
        logger.info("🔍 Loading trained model for feature verification...")

        model_dir = Path("models/artifacts")
        if not model_dir.exists():
            raise FileNotFoundError("No models/artifacts directory found")

        # Find latest model file
        model_files = list(model_dir.glob("*.pkl"))
        if not model_files:
            raise FileNotFoundError("No trained model files found")

        # Sort by timestamp and get latest
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading model: {latest_model}")

        with open(latest_model, 'rb') as f:
            model_data = pickle.load(f)

        if isinstance(model_data, dict) and 'model' in model_data:
            self.model = model_data['model']
        else:
            self.model = model_data

        # Extract feature information from model
        if hasattr(self.model, 'get_booster'):  # XGBoost model
            self.model_features = self.model.get_booster().feature_names
            logger.info(f"✅ XGBoost model loaded with {len(self.model_features)} features")
        elif hasattr(self.model, 'feature_names_in_'):  # sklearn model
            self.model_features = list(self.model.feature_names_in_)
            logger.info(f"✅ sklearn model loaded with {len(self.model_features)} features")
        else:
            logger.warning("⚠️ Cannot extract feature names from model")
            self.model_features = []

        return self.model_features

    def analyze_feature_generator_output(self):
        """Analyze what features the feature generator produces"""
        logger.info("🔍 Analyzing feature generator output...")

        # Generate sample features using the same logic as production
        sample_data = self._create_sample_ohlcv_data()
        sample_features = self.feature_generator.generate_symbol_features(
            sample_data, 'RELIANCE.NS', '2020-06-01', '2020-12-31'
        )

        if sample_features.empty:
            logger.error("❌ Feature generator produced no features")
            return []

        # Extract feature column names (exclude symbol and date)
        generated_features = [col for col in sample_features.columns
                            if col not in ['symbol', 'date']]

        logger.info(f"✅ Feature generator produces {len(generated_features)} features")
        logger.info(f"Sample features: {generated_features[:10]}...")

        return generated_features

    def _create_sample_ohlcv_data(self):
        """Create sample OHLCV data for feature generation testing"""
        dates = pd.date_range('2020-01-01', periods=300, freq='B')

        # Create realistic price series
        np.random.seed(42)
        base_price = 100
        prices = []

        for i in range(len(dates)):
            trend = i * 0.02
            cycle = 5 * np.sin(2 * np.pi * i / 100)
            noise = np.random.normal(0, 1.5)
            price = base_price + trend + cycle + noise
            prices.append(max(price, 1))

        return pd.DataFrame({
            'symbol': ['RELIANCE.NS'] * len(dates),
            'date': dates,
            'open': prices,
            'high': [p * np.random.uniform(1.005, 1.02) for p in prices],
            'low': [p * np.random.uniform(0.98, 0.995) for p in prices],
            'close': prices,
            'volume': [100000] * len(dates)
        })

    def perform_comprehensive_sync_check(self):
        """Perform comprehensive synchronization verification"""
        logger.info("🔬 STARTING COMPREHENSIVE FEATURE-MODEL SYNCHRONIZATION CHECK")
        logger.info("=" * 80)

        sync_results = {
            'feature_set_alignment': {},
            'feature_ordering': {},
            'data_types': {},
            'null_handling': {},
            'production_readiness': {}
        }

        # ================================
        # 1. LOAD MODEL AND EXTRACT FEATURES
        # ================================
        try:
            model_features = self.load_trained_model()
            sync_results['model_features'] = {
                'count': len(model_features),
                'names': model_features,
                'status': 'SUCCESS'
            }
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            sync_results['model_features'] = {
                'count': 0,
                'names': [],
                'status': 'FAILED',
                'error': str(e)
            }
            return sync_results

        # ================================
        # 2. ANALYZE FEATURE GENERATOR
        # ================================
        try:
            generated_features = self.analyze_feature_generator_output()
            sync_results['generated_features'] = {
                'count': len(generated_features),
                'names': generated_features,
                'status': 'SUCCESS'
            }
        except Exception as e:
            logger.error(f"❌ Failed to analyze feature generator: {e}")
            sync_results['generated_features'] = {
                'count': 0,
                'names': [],
                'status': 'FAILED',
                'error': str(e)
            }
            return sync_results

        # ================================
        # 3. FEATURE SET ALIGNMENT CHECK
        # ================================
        logger.info("\\n🔍 CHECKING FEATURE SET ALIGNMENT...")

        model_set = set(model_features)
        generated_set = set(generated_features)

        # Find common features
        common_features = model_set.intersection(generated_set)
        missing_in_generated = model_set - generated_set
        extra_in_generated = generated_set - model_set

        alignment_score = len(common_features) / len(model_set) if model_set else 0

        sync_results['feature_set_alignment'] = {
            'model_features_count': len(model_set),
            'generated_features_count': len(generated_set),
            'common_features_count': len(common_features),
            'missing_in_generated': list(missing_in_generated),
            'extra_in_generated': list(extra_in_generated),
            'alignment_score': alignment_score,
            'perfect_alignment': len(missing_in_generated) == 0 and len(extra_in_generated) == 0,
            'status': 'PERFECT' if alignment_score == 1.0 else 'PARTIAL' if alignment_score > 0.8 else 'POOR'
        }

        logger.info(f"📊 Feature Set Alignment: {alignment_score:.1%}")
        if missing_in_generated:
            logger.warning(f"⚠️ Missing features in generator: {list(missing_in_generated)[:5]}...")
        if extra_in_generated:
            logger.warning(f"⚠️ Extra features in generator: {list(extra_in_generated)[:5]}...")

        # ================================
        # 4. FEATURE ORDERING CHECK
        # ================================
        logger.info("\\n🔍 CHECKING FEATURE ORDERING...")

        # For models that care about feature ordering (like sklearn)
        if hasattr(self.model, 'feature_names_in_'):
            model_order = list(self.model.feature_names_in_)

            # Create ordered version of generated features matching model
            ordered_generated = []
            for feature in model_order:
                if feature in generated_features:
                    ordered_generated.append(feature)

            ordering_match = ordered_generated == model_order
            ordering_score = len(ordered_generated) / len(model_order)

            sync_results['feature_ordering'] = {
                'model_order': model_order,
                'generated_order': ordered_generated,
                'ordering_match': ordering_match,
                'ordering_score': ordering_score,
                'status': 'PERFECT' if ordering_match else 'ORDERED_SUBSET' if ordering_score == 1.0 else 'MISORDERED'
            }

            logger.info(f"📊 Feature Ordering: {'PERFECT' if ordering_match else f'{ordering_score:.1%} aligned'}")
        else:
            sync_results['feature_ordering'] = {
                'status': 'NOT_APPLICABLE',
                'note': 'Model does not require specific feature ordering'
            }
            logger.info("📊 Feature Ordering: Not applicable for this model type")

        # ================================
        # 5. PRODUCTION FEATURE MATRIX CREATION
        # ================================
        logger.info("\\n🔍 CREATING PRODUCTION FEATURE MATRIX...")

        # Load sample data and create production-ready feature matrix
        sample_data = self._create_sample_ohlcv_data()
        sample_features = self.feature_generator.generate_symbol_features(
            sample_data, 'RELIANCE.NS', '2020-06-01', '2020-12-31'
        )

        if not sample_features.empty:
            # Create feature matrix matching model expectations
            feature_cols = [col for col in sample_features.columns
                          if col not in ['symbol', 'date']]

            X_production = sample_features[feature_cols]

            # Align with model features if needed
            if self.model_features:
                # Ensure we have all required features in correct order
                aligned_features = []
                for feature in self.model_features:
                    if feature in X_production.columns:
                        aligned_features.append(feature)

                if aligned_features:
                    X_production = X_production[aligned_features]

            # ================================
            # 6. DATA TYPE VERIFICATION
            # ================================
            logger.info("\\n🔍 VERIFYING DATA TYPES...")

            dtype_issues = []
            for col in X_production.columns:
                dtype = X_production[col].dtype
                if dtype not in [np.float64, np.float32, 'float64', 'float32', int, 'int64', 'int32']:
                    dtype_issues.append(f"{col}: {dtype}")

            sync_results['data_types'] = {
                'numeric_columns': len(X_production.select_dtypes(include=[np.number]).columns),
                'total_columns': len(X_production.columns),
                'dtype_issues': dtype_issues,
                'all_numeric': len(dtype_issues) == 0,
                'status': 'GOOD' if len(dtype_issues) == 0 else 'ISSUES_FOUND'
            }

            logger.info(f"📊 Data Types: {len(dtype_issues)} issues found")

            # ================================
            # 7. NULL VALUE CHECK
            # ================================
            logger.info("\\n🔍 CHECKING FOR NULL VALUES...")

            null_counts = X_production.isnull().sum()
            null_columns = null_counts[null_counts > 0]

            total_nulls = null_counts.sum()
            null_percentage = total_nulls / (len(X_production) * len(X_production.columns)) * 100

            sync_results['null_handling'] = {
                'total_nulls': total_nulls,
                'null_percentage': null_percentage,
                'null_columns': list(null_columns.index) if len(null_columns) > 0 else [],
                'null_column_count': len(null_columns),
                'null_free': total_nulls == 0,
                'status': 'PERFECT' if total_nulls == 0 else 'HAS_NULLS'
            }

            logger.info(f"📊 Null Values: {total_nulls} total ({null_percentage:.2f}%)")

            # ================================
            # 8. MODEL PREDICTION TEST
            # ================================
            logger.info("\\n🔍 TESTING MODEL PREDICTIONS...")

            try:
                # Fill any remaining nulls for testing
                X_test = X_production.fillna(0)

                # Ensure we have the right features for the model
                if self.model_features and len(X_test.columns) == len(self.model_features):
                    predictions = self.model.predict_proba(X_test)[:, 1]
                    prediction_stats = {
                        'mean': float(np.mean(predictions)),
                        'std': float(np.std(predictions)),
                        'min': float(np.min(predictions)),
                        'max': float(np.max(predictions)),
                        'count': len(predictions)
                    }

                    sync_results['model_predictions'] = {
                        'status': 'SUCCESS',
                        'stats': prediction_stats
                    }

                    logger.info(f"✅ Model predictions successful: {len(predictions)} predictions generated")
                    logger.info(f"   Mean prediction: {prediction_stats['mean']:.3f}")
                else:
                    sync_results['model_predictions'] = {
                        'status': 'FEATURE_MISMATCH',
                        'message': f"Feature count mismatch: model expects {len(self.model_features)}, got {len(X_test.columns)}"
                    }
                    logger.error(f"❌ Feature count mismatch for predictions")

            except Exception as e:
                sync_results['model_predictions'] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                logger.error(f"❌ Model prediction test failed: {e}")

        else:
            logger.error("❌ No sample features generated")
            sync_results['production_matrix'] = {'status': 'FAILED', 'error': 'No features generated'}

        # ================================
        # 9. PRODUCTION READINESS ASSESSMENT
        # ================================
        logger.info("\\n🔍 ASSESSING PRODUCTION READINESS...")

        readiness_checks = {
            'feature_alignment': sync_results['feature_set_alignment']['perfect_alignment'],
            'no_nulls': sync_results['null_handling']['null_free'],
            'data_types_ok': sync_results['data_types']['all_numeric'],
            'model_predictable': sync_results.get('model_predictions', {}).get('status') == 'SUCCESS'
        }

        readiness_score = sum(readiness_checks.values()) / len(readiness_checks)

        sync_results['production_readiness'] = {
            'checks': readiness_checks,
            'readiness_score': readiness_score,
            'production_ready': readiness_score == 1.0,
            'status': 'PRODUCTION_READY' if readiness_score == 1.0 else 'NEEDS_FIXES'
        }

        logger.info(f"📊 Production Readiness: {readiness_score:.1%}")

        return sync_results

    def generate_sync_report(self, sync_results):
        """Generate comprehensive synchronization report"""
        logger.info("\\n" + "="*80)
        logger.info("📊 FEATURE-MODEL SYNCHRONIZATION REPORT")
        logger.info("="*80)

        # Executive Summary
        print("\\n📋 EXECUTIVE SUMMARY")
        print("-" * 50)

        readiness = sync_results.get('production_readiness', {})
        if readiness.get('production_ready', False):
            print("✅ SYNCHRONIZATION STATUS: PERFECT ALIGNMENT")
            print("🚀 SYSTEM READY FOR PRODUCTION DEPLOYMENT")
        else:
            print("⚠️ SYNCHRONIZATION STATUS: REQUIRES ATTENTION")
            print("🔧 FIXES NEEDED BEFORE PRODUCTION")

        print(f"Production Readiness Score: {readiness.get('readiness_score', 0):.1%}")

        # Detailed Results
        print("\\n🔍 SYNCHRONIZATION DETAILS")
        print("-" * 50)

        # Feature Set Alignment
        alignment = sync_results.get('feature_set_alignment', {})
        print(f"\\n📊 Feature Set Alignment:")
        print(f"   Model features: {alignment.get('model_features_count', 0)}")
        print(f"   Generated features: {alignment.get('generated_features_count', 0)}")
        print(f"   Common features: {alignment.get('common_features_count', 0)}")
        print(f"   Alignment score: {alignment.get('alignment_score', 0):.1%}")

        if alignment.get('missing_in_generated'):
            print(f"   ❌ Missing in generator: {len(alignment['missing_in_generated'])} features")
            print(f"      Sample: {alignment['missing_in_generated'][:3]}")

        if alignment.get('extra_in_generated'):
            print(f"   ⚠️ Extra in generator: {len(alignment['extra_in_generated'])} features")
            print(f"      Sample: {alignment['extra_in_generated'][:3]}")

        # Feature Ordering
        ordering = sync_results.get('feature_ordering', {})
        print(f"\\n📊 Feature Ordering:")
        print(f"   Status: {ordering.get('status', 'UNKNOWN')}")

        if ordering.get('status') in ['PERFECT', 'ORDERED_SUBSET']:
            print(f"   ✅ Feature ordering correct")
        elif ordering.get('status') == 'MISORDERED':
            print(f"   ⚠️ Feature ordering may cause issues")

        # Data Types
        dtypes = sync_results.get('data_types', {})
        print(f"\\n📊 Data Types:")
        print(f"   Numeric columns: {dtypes.get('numeric_columns', 0)}")
        print(f"   Total columns: {dtypes.get('total_columns', 0)}")

        if dtypes.get('dtype_issues'):
            print(f"   ❌ Type issues: {len(dtypes['dtype_issues'])} columns")
            print(f"      Sample: {dtypes['dtype_issues'][:2]}")
        else:
            print(f"   ✅ All columns numeric")

        # Null Handling
        nulls = sync_results.get('null_handling', {})
        print(f"\\n📊 Null Value Handling:")
        print(f"   Total nulls: {nulls.get('total_nulls', 0)}")
        print(f"   Null percentage: {nulls.get('null_percentage', 0):.2f}%")

        if nulls.get('null_free', False):
            print(f"   ✅ No null values detected")
        else:
            print(f"   ❌ Null values present in {nulls.get('null_column_count', 0)} columns")

        # Model Predictions
        predictions = sync_results.get('model_predictions', {})
        print(f"\\n📊 Model Predictions:")
        if predictions.get('status') == 'SUCCESS':
            stats = predictions.get('stats', {})
            print(f"   ✅ Predictions successful")
            print(f"   📈 Generated {stats.get('count', 0)} predictions")
            print(f"   📊 Mean prediction: {stats.get('mean', 0):.3f}")
        else:
            print(f"   ❌ Prediction test failed: {predictions.get('status', 'UNKNOWN')}")

        # Recommendations
        print("\\n🎯 SYNCHRONIZATION RECOMMENDATIONS")
        print("-" * 50)

        issues_found = []

        if not alignment.get('perfect_alignment', False):
            issues_found.append("Feature set misalignment - align feature generation with model expectations")

        if nulls.get('total_nulls', 0) > 0:
            issues_found.append("Null values detected - implement proper null handling")

        if dtypes.get('dtype_issues'):
            issues_found.append("Data type issues - ensure all features are numeric")

        if predictions.get('status') != 'SUCCESS':
            issues_found.append("Model prediction failures - fix feature synchronization")

        if issues_found:
            print("❌ Critical Issues Requiring Attention:")
            for i, issue in enumerate(issues_found, 1):
                print(f"   {i}. {issue}")
        else:
            print("✅ All synchronization checks passed!")
            print("🚀 System is ready for production deployment")

        # Save detailed report
        import json
        report_file = f"reports/feature_model_sync_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(sync_results, f, indent=2, default=str)

        print(f"\\n💾 Detailed synchronization report saved to: {report_file}")

        return sync_results

def main():
    """Main synchronization verification function"""
    print("🔄 FEATURE-MODEL SYNCHRONIZATION VERIFICATION")
    print("Ensuring perfect alignment for production ML trading")
    print("=" * 80)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/feature_model_sync.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Initialize synchronizer
        synchronizer = FeatureModelSynchronizer()

        # Perform comprehensive sync check
        sync_results = synchronizer.perform_comprehensive_sync_check()

        # Generate detailed report
        synchronizer.generate_sync_report(sync_results)

        # Final status
        readiness = sync_results.get('production_readiness', {})
        if readiness.get('production_ready', False):
            print("\\n🎉 SYNCHRONIZATION SUCCESSFUL!")
            print("✅ Perfect feature-model alignment achieved")
            print("🚀 Ready for quantitative audit and production deployment")
            return 0
        else:
            print("\\n⚠️ SYNCHRONIZATION ISSUES DETECTED")
            print("🔧 Address the issues above before proceeding")
            return 1

    except Exception as e:
        logger.error(f"Synchronization verification failed: {e}", exc_info=True)
        print(f"❌ Synchronization verification failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
