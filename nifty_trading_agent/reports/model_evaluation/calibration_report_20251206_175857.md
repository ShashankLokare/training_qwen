# Model Calibration Analysis Report

**Generated:** 2025-12-06 17:58:57
**Brier Score:** 0.1190

## Overview

This report analyzes the calibration of the trading signal prediction model.
Calibration measures how well predicted probabilities match actual outcomes.

- **Brier Score**: 0.1190 (lower is better, 0.0 = perfect calibration)
- **Analysis Period**: 2025-11-26 to 2025-12-31

## Calibration Table

| Probability Bucket | Count | Avg Predicted | Actual Rate | Calibration Error |
|-------------------|-------|---------------|-------------|-------------------|
|0.0-0.5|280|0.343|0.004|0.340|


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
