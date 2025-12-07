#!/usr/bin/env python3
"""
Feature Validation Utilities for Nifty Trading Agent v2
Ensures feature-model alignment and validates runtime features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class FeatureValidator:
    """
    Validates feature alignment between model expectations and runtime data
    """

    def __init__(self):
        """Initialize feature validator"""
        pass

    def validate_feature_alignment(self, model_feature_list: List[str],
                                 runtime_features: pd.DataFrame,
                                 strict: bool = True) -> Dict[str, Any]:
        """
        Validate that runtime features match model expectations

        Args:
            model_feature_list: List of feature names expected by the model
            runtime_features: DataFrame with runtime features
            strict: If True, raise exception on mismatch; if False, return validation results

        Returns:
            Dictionary with validation results

        Raises:
            ValueError: If strict=True and validation fails
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_features': [],
            'extra_features': [],
            'dtype_mismatches': [],
            'feature_count_match': True,
            'order_match': True
        }

        # Check 1: All required features present
        runtime_feature_cols = list(runtime_features.columns)
        missing_features = [feat for feat in model_feature_list if feat not in runtime_feature_cols]

        if missing_features:
            error_msg = f"Missing required features: {missing_features}"
            validation_results['errors'].append(error_msg)
            validation_results['missing_features'] = missing_features
            validation_results['is_valid'] = False

        # Check 2: No unexpected extra features (warn only)
        extra_features = [feat for feat in runtime_feature_cols if feat not in model_feature_list]
        if extra_features:
            warning_msg = f"Extra features found (will be ignored): {extra_features}"
            validation_results['warnings'].append(warning_msg)
            validation_results['extra_features'] = extra_features

        # Check 3: Feature count matches
        if len(model_feature_list) != len(runtime_feature_cols):
            validation_results['feature_count_match'] = False
            if len(missing_features) == 0:  # Only if no missing features
                validation_results['warnings'].append(
                    f"Feature count mismatch: expected {len(model_feature_list)}, got {len(runtime_feature_cols)}"
                )

        # Check 4: Feature order matches (if counts match)
        if validation_results['feature_count_match'] and len(missing_features) == 0:
            if model_feature_list != runtime_feature_cols:
                validation_results['order_match'] = False
                validation_results['warnings'].append("Feature order does not match expected order")

        # Check 5: Data types are numeric
        non_numeric_features = []
        for col in runtime_feature_cols:
            if col in model_feature_list:  # Only check expected features
                dtype = runtime_features[col].dtype
                if not pd.api.types.is_numeric_dtype(dtype):
                    non_numeric_features.append(f"{col} ({dtype})")

        if non_numeric_features:
            error_msg = f"Non-numeric features found: {non_numeric_features}"
            validation_results['errors'].append(error_msg)
            validation_results['dtype_mismatches'] = non_numeric_features
            validation_results['is_valid'] = False

        # Check 6: No NaN or infinite values in features
        for col in runtime_feature_cols:
            if col in model_feature_list:
                col_data = runtime_features[col]
                nan_count = col_data.isna().sum()
                inf_count = np.isinf(col_data).sum()

                if nan_count > 0:
                    validation_results['warnings'].append(
                        f"Feature '{col}' contains {nan_count} NaN values"
                    )

                if inf_count > 0:
                    validation_results['warnings'].append(
                        f"Feature '{col}' contains {inf_count} infinite values"
                    )

        # Log results
        self._log_validation_results(validation_results)

        # Handle strict mode
        if strict and not validation_results['is_valid']:
            error_summary = "; ".join(validation_results['errors'])
            raise ValueError(f"Feature validation failed: {error_summary}")

        return validation_results

    def _log_validation_results(self, results: Dict[str, Any]):
        """Log validation results"""
        if results['is_valid']:
            logger.info("✅ Feature validation passed")
        else:
            logger.error("❌ Feature validation failed")

        for error in results['errors']:
            logger.error(f"  Error: {error}")

        for warning in results['warnings']:
            logger.warning(f"  Warning: {warning}")

    def get_feature_stats(self, features_df: pd.DataFrame,
                         feature_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get statistics for features to help with debugging

        Args:
            features_df: DataFrame with features
            feature_list: Optional list of features to analyze

        Returns:
            Dictionary with feature statistics
        """
        if feature_list is None:
            feature_list = list(features_df.columns)

        stats = {
            'total_features': len(feature_list),
            'feature_names': feature_list[:10],  # First 10 for brevity
            'data_types': {},
            'missing_values': {},
            'infinite_values': {},
            'value_ranges': {}
        }

        for col in feature_list:
            if col in features_df.columns:
                col_data = features_df[col]

                # Data type
                stats['data_types'][col] = str(col_data.dtype)

                # Missing values
                nan_count = col_data.isna().sum()
                stats['missing_values'][col] = int(nan_count)

                # Infinite values
                inf_count = np.isinf(col_data).sum()
                stats['infinite_values'][col] = int(inf_count)

                # Value ranges (for numeric columns)
                if pd.api.types.is_numeric_dtype(col_data):
                    stats['value_ranges'][col] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std())
                    }

        return stats

    def suggest_feature_fixes(self, validation_results: Dict[str, Any]) -> List[str]:
        """
        Suggest fixes for validation issues

        Args:
            validation_results: Results from validate_feature_alignment

        Returns:
            List of suggested fixes
        """
        suggestions = []

        if validation_results['missing_features']:
            suggestions.append(
                f"Add missing features: {validation_results['missing_features'][:5]}..."
                if len(validation_results['missing_features']) > 5
                else f"Add missing features: {validation_results['missing_features']}"
            )

        if validation_results['dtype_mismatches']:
            suggestions.append(
                f"Convert non-numeric features to numeric: {validation_results['dtype_mismatches'][:3]}..."
                if len(validation_results['dtype_mismatches']) > 3
                else f"Convert non-numeric features to numeric: {validation_results['dtype_mismatches']}"
            )

        if not validation_results['order_match']:
            suggestions.append("Reorder features to match model expectations")

        if validation_results['extra_features']:
            suggestions.append("Remove extra features or ensure they are handled properly")

        # Check for NaN/infinite warnings
        nan_warnings = [w for w in validation_results['warnings'] if 'NaN' in w]
        if nan_warnings:
            suggestions.append("Handle missing values (consider imputation or removal)")

        inf_warnings = [w for w in validation_results['warnings'] if 'infinite' in w]
        if inf_warnings:
            suggestions.append("Handle infinite values (consider clipping or removal)")

        if not suggestions:
            suggestions.append("No issues detected - features look good!")

        return suggestions

    def create_feature_report(self, model_features: List[str],
                            runtime_features: pd.DataFrame,
                            output_path: Optional[str] = None) -> str:
        """
        Create a detailed feature validation report

        Args:
            model_features: Expected feature names
            runtime_features: Runtime feature DataFrame
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        validation = self.validate_feature_alignment(model_features, runtime_features, strict=False)
        stats = self.get_feature_stats(runtime_features, model_features)
        suggestions = self.suggest_feature_fixes(validation)

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("FEATURE VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Overall status
        status = "✅ PASSED" if validation['is_valid'] else "❌ FAILED"
        report_lines.append(f"Overall Status: {status}")
        report_lines.append("")

        # Summary
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Expected features: {len(model_features)}")
        report_lines.append(f"  Runtime features: {len(runtime_features.columns)}")
        report_lines.append(f"  Missing features: {len(validation['missing_features'])}")
        report_lines.append(f"  Extra features: {len(validation['extra_features'])}")
        report_lines.append(f"  Feature order match: {validation['order_match']}")
        report_lines.append("")

        # Errors
        if validation['errors']:
            report_lines.append("ERRORS:")
            for error in validation['errors']:
                report_lines.append(f"  ❌ {error}")
            report_lines.append("")

        # Warnings
        if validation['warnings']:
            report_lines.append("WARNINGS:")
            for warning in validation['warnings']:
                report_lines.append(f"  ⚠️  {warning}")
            report_lines.append("")

        # Suggestions
        if suggestions:
            report_lines.append("SUGGESTED FIXES:")
            for suggestion in suggestions:
                report_lines.append(f"  💡 {suggestion}")
            report_lines.append("")

        # Feature statistics (first 10)
        report_lines.append("FEATURE STATISTICS (first 10):")
        for i, feature in enumerate(model_features[:10]):
            if feature in stats['data_types']:
                dtype = stats['data_types'][feature]
                missing = stats['missing_values'].get(feature, 0)
                infinite = stats['infinite_values'].get(feature, 0)
                report_lines.append(f"  {feature}:")
                report_lines.append(f"    Type: {dtype}, Missing: {missing}, Infinite: {infinite}")

                if feature in stats['value_ranges']:
                    vr = stats['value_ranges'][feature]
                    report_lines.append(f"    Range: [{vr['min']:.3f}, {vr['max']:.3f}], Mean: {vr['mean']:.3f}")
        report_lines.append("")

        report = "\n".join(report_lines)

        # Save to file if requested
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(report)
                logger.info(f"Feature validation report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save report to {output_path}: {e}")

        return report

def validate_model_features(model_path: str, features_df: pd.DataFrame,
                          strict: bool = True) -> bool:
    """
    Convenience function to validate features for a saved model

    Args:
        model_path: Path to saved model pickle file
        features_df: Runtime features DataFrame
        strict: Whether to raise exceptions on failure

    Returns:
        True if validation passes, False otherwise
    """
    import pickle

    try:
        # Load model metadata
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        if 'metadata' not in model_data:
            logger.error("Model file does not contain metadata")
            if strict:
                raise ValueError("Model file does not contain metadata")
            return False

        metadata = model_data['metadata']
        model_features = metadata.get('feature_names', [])

        if not model_features:
            logger.error("Model metadata does not contain feature names")
            if strict:
                raise ValueError("Model metadata does not contain feature names")
            return False

        # Validate features
        validator = FeatureValidator()
        validation_results = validator.validate_feature_alignment(
            model_features, features_df, strict=strict
        )

        return validation_results['is_valid']

    except Exception as e:
        logger.error(f"Failed to validate model features: {e}")
        if strict:
            raise
        return False

def main():
    """Demo function for feature validation"""
    print("🔍 FEATURE VALIDATION UTILITIES - DEMO")
    print("=" * 45)

    # Create sample model features
    model_features = ['feature_1', 'feature_2', 'feature_3', 'feature_4']

    # Create sample runtime features (with some issues)
    np.random.seed(42)
    n_samples = 100

    runtime_data = {
        'feature_1': np.random.normal(0, 1, n_samples),  # Good
        'feature_2': np.random.normal(0, 1, n_samples),  # Good
        'feature_3': ['string_value'] * n_samples,       # Wrong type
        'extra_feature': np.random.normal(0, 1, n_samples),  # Extra
        # Missing feature_4
    }

    runtime_df = pd.DataFrame(runtime_data)

    # Validate
    validator = FeatureValidator()
    results = validator.validate_feature_alignment(model_features, runtime_df, strict=False)

    print(f"Validation passed: {results['is_valid']}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Warnings: {len(results['warnings'])}")

    # Print suggestions
    suggestions = validator.suggest_feature_fixes(results)
    print("\nSuggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")

    # Create report
    report = validator.create_feature_report(model_features, runtime_df)
    print(f"\nReport preview (first 500 chars):\n{report[:500]}...")

if __name__ == "__main__":
    main()
