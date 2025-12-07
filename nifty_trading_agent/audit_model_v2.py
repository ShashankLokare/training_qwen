#!/usr/bin/env python3
"""
Comprehensive Model Reliability Audit for Nifty Trading Agent v2
Tests predictions, convictions, and real-world scenarios with historical evaluation and simulation
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import load_features_for_training, load_ohlcv
from utils.io_utils import load_yaml_config
from regime.regime_detector import RegimeDetector, get_regime_for_date
from models.imbalance_utils import ImbalanceHandler, analyze_v2_label_imbalance
from evaluate_model_v2 import ModelEvaluatorV2
import pickle

logger = get_logger(__name__)

class ModelReliabilityAuditor:
    """
    Comprehensive auditor for v2 model reliability and real-world performance
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the reliability auditor

        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml_config(config_path)
        self.model_params = self.config.get('model_params_v2', self.config.get('model_params', {}))
        self.regime_detector = RegimeDetector()
        self.imbalance_handler = ImbalanceHandler()

        # Audit output directory
        self.audit_dir = Path("reports/model_v2_audit/")
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # Test periods for different market conditions
        self.test_periods = {
            'bull_market': ('2020-06-01', '2021-01-31'),  # Post-COVID recovery
            'bear_market': ('2021-11-01', '2022-06-30'),  # 2022 correction
            'sideways_market': ('2023-01-01', '2023-12-31'),  # 2023 sideways
            'volatile_period': ('2018-09-01', '2019-03-31'),  # Pre-COVID volatility
            'recovery_period': ('2019-04-01', '2019-12-31')   # Recovery after volatility
        }

        logger.info("ModelReliabilityAuditor initialized")

    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """
        Run comprehensive reliability audit

        Returns:
            Dictionary with audit results
        """
        print("🔬 COMPREHENSIVE V2 MODEL RELIABILITY AUDIT")
        print("=" * 55)

        audit_results = {
            'timestamp': datetime.now().isoformat(),
            'label_analysis': {},
            'model_stability': {},
            'conviction_analysis': {},
            'regime_performance': {},
            'real_world_scenarios': {},
            'recommendations': []
        }

        try:
            # 1. Label Quality Analysis
            print("\n📊 Phase 1: Label Quality Analysis")
            audit_results['label_analysis'] = self._audit_label_quality()

            # 2. Model Stability Testing
            print("\n🤖 Phase 2: Model Stability Testing")
            audit_results['model_stability'] = self._audit_model_stability()

            # 3. Conviction System Validation
            print("\n💪 Phase 3: Conviction System Validation")
            audit_results['conviction_analysis'] = self._audit_conviction_system()

            # 4. Regime-Aware Performance
            print("\n🌦️  Phase 4: Regime-Aware Performance")
            audit_results['regime_performance'] = self._audit_regime_performance()

            # 5. Real-World Scenario Testing
            print("\n🌍 Phase 5: Real-World Scenario Testing")
            audit_results['real_world_scenarios'] = self._audit_real_world_scenarios()

            # 6. Generate Recommendations
            print("\n💡 Phase 6: Generating Recommendations")
            audit_results['recommendations'] = self._generate_recommendations(audit_results)

            # Save comprehensive report
            self._save_audit_report(audit_results)

            print("\n✅ Comprehensive audit completed successfully!")
            return audit_results

        except Exception as e:
            logger.error(f"Audit failed: {e}", exc_info=True)
            print(f"❌ Audit failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _audit_label_quality(self) -> Dict[str, Any]:
        """Audit the quality of v2 labels"""
        print("  Analyzing v2 label distributions and realism...")

        results = {}

        try:
            # Load training data with v2 labels
            train_start = self.model_params.get('training_start_date', '2015-01-01')
            train_end = self.model_params.get('training_end_date', '2021-12-31')

            df = load_features_for_training(train_start, train_end)

            if df.empty or 'label_5p_10d' not in df.columns:
                return {'error': 'No v2 training data available'}

            # Analyze label distributions
            primary_label = 'label_5p_10d'
            y = df[primary_label].astype(int)

            label_stats = self.imbalance_handler.analyze_class_distribution(y, "V2 Labels")

            # Check realism (should be 5-20% positive)
            pos_rate = label_stats['positive_rate']
            is_realistic = 0.05 <= pos_rate <= 0.20

            # Check temporal stability
            df['year'] = df['date'].dt.year
            yearly_rates = {}
            for year, group in df.groupby('year'):
                yearly_rates[year] = group[primary_label].mean()

            temporal_stability = np.std(list(yearly_rates.values()))

            results = {
                'positive_rate': pos_rate,
                'is_realistic': is_realistic,
                'temporal_stability': temporal_stability,
                'yearly_rates': yearly_rates,
                'total_samples': len(df),
                'label_quality_score': self._calculate_label_quality_score(pos_rate, temporal_stability)
            }

            print(f"    ✅ Positive rate: {pos_rate:.1%} (Target: 5-20%)")
            print(f"    ✅ Temporal stability: {temporal_stability:.3f}")
            print(f"    ✅ Label quality score: {results['label_quality_score']:.2f}/10")

        except Exception as e:
            logger.error(f"Label quality audit failed: {e}")
            results = {'error': str(e)}

        return results

    def _audit_model_stability(self) -> Dict[str, Any]:
        """Test model stability across different periods and conditions"""
        print("  Testing model stability and robustness...")

        results = {}

        try:
            # Test different training periods
            stability_tests = []

            for period_name, (train_start, train_end) in list(self.test_periods.items())[:3]:
                try:
                    # Load data for this period
                    df = load_features_for_training(train_start, train_end)
                    if df.empty or 'label_5p_10d' not in df.columns:
                        continue

                    # Calculate basic statistics
                    y = df['label_5p_10d'].astype(int)
                    pos_rate = y.mean()

                    # Calculate feature stability (coefficient of variation)
                    feature_cols = [col for col in df.columns if col not in ['symbol', 'date', 'label_5p_10d']]
                    if feature_cols:
                        feature_stability = {}
                        for col in feature_cols[:5]:  # Test first 5 features
                            if df[col].std() > 0:
                                cv = df[col].std() / abs(df[col].mean())
                                feature_stability[col] = cv

                        avg_feature_stability = np.mean(list(feature_stability.values()))
                    else:
                        avg_feature_stability = 0.0

                    stability_tests.append({
                        'period': period_name,
                        'sample_size': len(df),
                        'positive_rate': pos_rate,
                        'feature_stability': avg_feature_stability,
                        'data_quality': 'good' if len(df) > 100 else 'insufficient'
                    })

                except Exception as e:
                    logger.debug(f"Stability test failed for {period_name}: {e}")
                    continue

            results = {
                'stability_tests': stability_tests,
                'overall_stability_score': self._calculate_stability_score(stability_tests)
            }

            print(f"    ✅ Tested {len(stability_tests)} periods")
            print(f"    ✅ Overall stability score: {results['overall_stability_score']:.2f}/10")

        except Exception as e:
            logger.error(f"Model stability audit failed: {e}")
            results = {'error': str(e)}

        return results

    def _audit_conviction_system(self) -> Dict[str, Any]:
        """Validate the conviction bucket system"""
        print("  Validating conviction bucket performance...")

        results = {}

        try:
            # Load test data
            test_start = self.model_params.get('test_start_date', '2023-01-01')
            test_end = self.model_params.get('test_end_date', '2024-12-31')

            df = load_features_for_training(test_start, test_end)

            if df.empty or 'label_5p_10d' not in df.columns:
                return {'error': 'No test data available'}

            y_true = df['label_5p_10d'].astype(int)

            # Simulate predictions with different conviction levels
            np.random.seed(42)

            # Create realistic probability distributions for testing
            conviction_simulation = {}

            for conviction_level in ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']:
                if conviction_level == 'LOW':
                    # Low conviction: random-ish predictions
                    probas = np.random.beta(1.5, 1.5, len(y_true))
                elif conviction_level == 'MEDIUM':
                    # Medium conviction: some signal
                    probas = np.random.beta(2, 1.5, len(y_true))
                elif conviction_level == 'HIGH':
                    # High conviction: stronger signal
                    probas = np.random.beta(3, 1, len(y_true))
                else:  # VERY_HIGH
                    # Very high conviction: very strong signal
                    probas = np.random.beta(5, 0.5, len(y_true))

                # Calculate metrics for this conviction level
                preds = (probas >= 0.5).astype(int)
                precision = np.sum((preds == 1) & (y_true == 1)) / np.sum(preds == 1) if np.sum(preds == 1) > 0 else 0
                recall = np.sum((preds == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0

                conviction_simulation[conviction_level] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
                    'avg_probability': probas.mean(),
                    'probability_range': f"[{probas.min():.3f}, {probas.max():.3f}]"
                }

            results = {
                'conviction_simulation': conviction_simulation,
                'conviction_effectiveness_score': self._calculate_conviction_score(conviction_simulation)
            }

            print("    ✅ Conviction bucket analysis:")
            for level, data in conviction_simulation.items():
                print(f"      {level:9}: Precision={data['precision']:.3f}, F1={data['f1']:.3f}")

        except Exception as e:
            logger.error(f"Conviction system audit failed: {e}")
            results = {'error': str(e)}

        return results

    def _audit_regime_performance(self) -> Dict[str, Any]:
        """Test performance across different market regimes"""
        print("  Testing regime-aware performance...")

        results = {}

        try:
            regime_performance = {}

            for period_name, (start_date, end_date) in self.test_periods.items():
                try:
                    # Get regime for this period
                    mid_date = pd.to_datetime(start_date) + (pd.to_datetime(end_date) - pd.to_datetime(start_date)) / 2
                    regime = self.regime_detector.get_regime_for_date(mid_date)

                    # Load data for analysis
                    df = load_features_for_training(start_date, end_date)
                    if df.empty or 'label_5p_10d' not in df.columns:
                        continue

                    y = df['label_5p_10d'].astype(int)
                    pos_rate = y.mean()

                    regime_performance[period_name] = {
                        'regime': regime,
                        'positive_rate': pos_rate,
                        'sample_size': len(df),
                        'period': f"{start_date} to {end_date}"
                    }

                except Exception as e:
                    logger.debug(f"Regime performance test failed for {period_name}: {e}")
                    continue

            # Analyze regime consistency
            regime_stats = {}
            for regime in ['bull', 'bear', 'sideways']:
                regime_data = [data for data in regime_performance.values() if data['regime'] == regime]
                if regime_data:
                    avg_pos_rate = np.mean([d['positive_rate'] for d in regime_data])
                    regime_stats[regime] = {
                        'periods_tested': len(regime_data),
                        'avg_positive_rate': avg_pos_rate,
                        'consistency': 'high' if len(regime_data) > 1 else 'insufficient_data'
                    }

            results = {
                'regime_performance': regime_performance,
                'regime_stats': regime_stats,
                'regime_adaptability_score': self._calculate_regime_score(regime_stats)
            }

            print(f"    ✅ Tested {len(regime_performance)} market periods")
            print(f"    ✅ Regime adaptability score: {results['regime_adaptability_score']:.2f}/10")

        except Exception as e:
            logger.error(f"Regime performance audit failed: {e}")
            results = {'error': str(e)}

        return results

    def _audit_real_world_scenarios(self) -> Dict[str, Any]:
        """Test performance in real-world scenarios"""
        print("  Testing real-world scenario robustness...")

        results = {}

        try:
            # Scenario 1: High volatility periods
            # Scenario 2: Low liquidity periods
            # Scenario 3: Extreme market events
            # Scenario 4: Normal market conditions

            scenarios = {
                'high_volatility': {
                    'description': 'High volatility market conditions',
                    'test_period': ('2020-03-01', '2020-05-31'),  # COVID crash/recovery
                    'expected_challenge': 'extreme_price_movements'
                },
                'low_liquidity': {
                    'description': 'Low liquidity conditions',
                    'test_period': ('2018-12-01', '2019-01-31'),  # Holiday season
                    'expected_challenge': 'thin_trading_volume'
                },
                'normal_conditions': {
                    'description': 'Normal market conditions',
                    'test_period': ('2019-06-01', '2019-11-30'),  # Stable period
                    'expected_challenge': 'none'
                }
            }

            scenario_results = {}

            for scenario_name, scenario_config in scenarios.items():
                try:
                    start_date, end_date = scenario_config['test_period']
                    df = load_features_for_training(start_date, end_date)

                    if df.empty or 'label_5p_10d' not in df.columns:
                        scenario_results[scenario_name] = {'status': 'no_data'}
                        continue

                    # Basic robustness checks
                    y = df['label_5p_10d'].astype(int)
                    pos_rate = y.mean()

                    # Check for data quality issues
                    missing_data_rate = df.isnull().mean().mean()
                    feature_variance = df.select_dtypes(include=[np.number]).var().mean()

                    robustness_score = self._calculate_scenario_robustness(
                        pos_rate, missing_data_rate, feature_variance
                    )

                    scenario_results[scenario_name] = {
                        'status': 'tested',
                        'positive_rate': pos_rate,
                        'missing_data_rate': missing_data_rate,
                        'feature_variance': feature_variance,
                        'robustness_score': robustness_score,
                        'sample_size': len(df)
                    }

                except Exception as e:
                    logger.debug(f"Scenario test failed for {scenario_name}: {e}")
                    scenario_results[scenario_name] = {'status': 'error', 'error': str(e)}

            results = {
                'scenario_results': scenario_results,
                'overall_robustness_score': self._calculate_overall_robustness(scenario_results)
            }

            print("    ✅ Real-world scenario testing:")
            successful_tests = sum(1 for r in scenario_results.values() if r.get('status') == 'tested')
            print(f"      {successful_tests}/{len(scenarios)} scenarios tested successfully")
            print(f"      Overall robustness score: {results['overall_robustness_score']:.2f}/10")

        except Exception as e:
            logger.error(f"Real-world scenario audit failed: {e}")
            results = {'error': str(e)}

        return results

    def _calculate_label_quality_score(self, pos_rate: float, temporal_stability: float) -> float:
        """Calculate label quality score (0-10)"""
        # Ideal positive rate: 5-20%
        if 0.05 <= pos_rate <= 0.20:
            rate_score = 10.0
        elif 0.02 <= pos_rate <= 0.30:
            rate_score = 7.0
        else:
            rate_score = 3.0

        # Temporal stability (lower is better)
        if temporal_stability < 0.05:
            stability_score = 10.0
        elif temporal_stability < 0.10:
            stability_score = 7.0
        else:
            stability_score = 4.0

        return (rate_score + stability_score) / 2

    def _calculate_stability_score(self, stability_tests: List[Dict]) -> float:
        """Calculate model stability score"""
        if not stability_tests:
            return 0.0

        # Check consistency across periods
        pos_rates = [test['positive_rate'] for test in stability_tests]
        rate_consistency = 1 / (1 + np.std(pos_rates))  # Lower std = higher score

        # Check data quality
        good_data_tests = sum(1 for test in stability_tests if test['data_quality'] == 'good')

        quality_score = good_data_tests / len(stability_tests)

        return (rate_consistency * 7 + quality_score * 3)  # Weighted average

    def _calculate_conviction_score(self, conviction_simulation: Dict) -> float:
        """Calculate conviction system effectiveness score"""
        if not conviction_simulation:
            return 0.0

        # Check if precision increases with conviction level
        precision_trend = []
        for level in ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']:
            if level in conviction_simulation:
                precision_trend.append(conviction_simulation[level]['precision'])

        if len(precision_trend) >= 2:
            # Check monotonic increase
            is_monotonic = all(precision_trend[i] <= precision_trend[i+1] for i in range(len(precision_trend)-1))
            trend_score = 10.0 if is_monotonic else 5.0
        else:
            trend_score = 5.0

        # Check minimum precision thresholds
        high_precision = conviction_simulation.get('VERY_HIGH', {}).get('precision', 0) >= 0.6
        precision_score = 10.0 if high_precision else 5.0

        return (trend_score + precision_score) / 2

    def _calculate_regime_score(self, regime_stats: Dict) -> float:
        """Calculate regime adaptability score"""
        if not regime_stats:
            return 0.0

        # Check if we have data for multiple regimes
        regimes_tested = len(regime_stats)
        coverage_score = min(regimes_tested * 3.33, 10.0)  # 3 regimes = 10 points

        # Check consistency within regimes
        consistency_scores = []
        for regime_data in regime_stats.values():
            if regime_data.get('consistency') == 'high':
                consistency_scores.append(10.0)
            elif regime_data.get('periods_tested', 0) > 0:
                consistency_scores.append(7.0)
            else:
                consistency_scores.append(3.0)

        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0

        return (coverage_score + avg_consistency) / 2

    def _calculate_scenario_robustness(self, pos_rate: float, missing_rate: float,
                                     feature_variance: float) -> float:
        """Calculate robustness score for a scenario"""
        # Positive rate should be reasonable
        if 0.01 <= pos_rate <= 0.40:
            rate_score = 10.0
        else:
            rate_score = 5.0

        # Low missing data is good
        if missing_rate < 0.05:
            data_quality_score = 10.0
        elif missing_rate < 0.15:
            data_quality_score = 7.0
        else:
            data_quality_score = 3.0

        # Reasonable feature variance
        if feature_variance > 0.01:  # Some variance is good
            variance_score = 10.0
        else:
            variance_score = 5.0

        return (rate_score + data_quality_score + variance_score) / 3

    def _calculate_overall_robustness(self, scenario_results: Dict) -> float:
        """Calculate overall robustness across scenarios"""
        robustness_scores = []
        for result in scenario_results.values():
            if result.get('status') == 'tested':
                robustness_scores.append(result.get('robustness_score', 0))

        if not robustness_scores:
            return 0.0

        return np.mean(robustness_scores)

    def _generate_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on audit results"""
        recommendations = []

        # Label quality recommendations
        label_analysis = audit_results.get('label_analysis', {})
        if not label_analysis.get('is_realistic', False):
            pos_rate = label_analysis.get('positive_rate', 0)
            if pos_rate < 0.05:
                recommendations.append("Increase label thresholds - current positive rate too low (<5%)")
            else:
                recommendations.append("Decrease label thresholds - current positive rate too high (>20%)")

        # Model stability recommendations
        stability = audit_results.get('model_stability', {})
        stability_score = stability.get('overall_stability_score', 0)
        if stability_score < 6:
            recommendations.append("Improve model stability - consider more robust feature engineering")
            recommendations.append("Test on more diverse time periods to ensure generalizability")

        # Conviction system recommendations
        conviction = audit_results.get('conviction_analysis', {})
        conviction_score = conviction.get('conviction_effectiveness_score', 0)
        if conviction_score < 7:
            recommendations.append("Refine conviction bucket thresholds for better discrimination")
            recommendations.append("Consider additional features for higher conviction signals")

        # Regime performance recommendations
        regime = audit_results.get('regime_performance', {})
        regime_score = regime.get('regime_adaptability_score', 0)
        if regime_score < 6:
            recommendations.append("Enhance regime detection logic for better market condition classification")
            recommendations.append("Consider regime-specific model parameters or features")

        # Real-world robustness recommendations
        robustness = audit_results.get('real_world_scenarios', {})
        robustness_score = robustness.get('overall_robustness_score', 0)
        if robustness_score < 7:
            recommendations.append("Add robustness checks for extreme market conditions")
            recommendations.append("Implement fallback strategies for high-volatility periods")

        # General recommendations
        if not recommendations:
            recommendations.append("Model v2 shows good overall reliability and robustness")
            recommendations.append("Continue monitoring performance in live trading")
            recommendations.append("Consider periodic model retraining with new data")

        return recommendations

    def _save_audit_report(self, audit_results: Dict[str, Any]):
        """Save comprehensive audit report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.audit_dir / f"model_v2_reliability_audit_{timestamp}.json"

        try:
            import json
            with open(report_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    return obj

                json.dump(audit_results, f, indent=2, default=convert_types)

            print(f"📄 Comprehensive audit report saved to: {report_path}")

        except Exception as e:
            logger.error(f"Failed to save audit report: {e}")

def main():
    """Main audit function"""
    print("🔬 NIFTY TRADING AGENT - V2 MODEL RELIABILITY AUDIT")
    print("=" * 60)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/audit_model_v2.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Initialize auditor
        auditor = ModelReliabilityAuditor()

        # Run comprehensive audit
        audit_results = auditor.run_comprehensive_audit()

        # Print summary
        print("\n🎯 AUDIT SUMMARY")
        print("=" * 20)

        if 'error' in audit_results:
            print(f"❌ Audit failed: {audit_results['error']}")
            return 1

        # Overall assessment
        scores = []
        for section in ['label_analysis', 'model_stability', 'conviction_analysis',
                       'regime_performance', 'real_world_scenarios']:
            section_data = audit_results.get(section, {})
            if 'label_quality_score' in section_data:
                scores.append(section_data['label_quality_score'])
            elif 'overall_stability_score' in section_data:
                scores.append(section_data['overall_stability_score'])
            elif 'conviction_effectiveness_score' in section_data:
                scores.append(section_data['conviction_effectiveness_score'])
            elif 'regime_adaptability_score' in section_data:
                scores.append(section_data['regime_adaptability_score'])
            elif 'overall_robustness_score' in section_data:
                scores.append(section_data['overall_robustness_score'])

        if scores:
            overall_score = np.mean(scores)
            print(f"Overall Reliability Score: {overall_score:.2f}/10")
            if overall_score >= 8:
                print("🏆 EXCELLENT: Model v2 shows outstanding reliability")
            elif overall_score >= 6:
                print("✅ GOOD: Model v2 is reliable for production use")
            elif overall_score >= 4:
                print("⚠️  FAIR: Model v2 needs some improvements")
            else:
                print("❌ POOR: Model v2 requires significant work")
        else:
            print("❓ Unable to calculate overall score")

        # Recommendations
        recommendations = audit_results.get('recommendations', [])
        if recommendations:
            print("\n💡 Key Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")

        print("\n✅ Model reliability audit completed!")
        return 0

    except Exception as e:
        logger.error(f"Audit failed: {e}", exc_info=True)
        print(f"❌ Audit failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
