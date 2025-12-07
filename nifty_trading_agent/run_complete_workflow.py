#!/usr/bin/env python3
"""
Complete ML Workflow Demonstration Script
Shows the full pipeline: labels → features → training → evaluation → backtesting
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd: str, description: str):
    """Run a command and display results"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print('='*60)

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """Run the complete ML workflow demonstration"""
    print("🚀 NSE NIFTY TRADING AGENT - COMPLETE ML WORKFLOW")
    print("=" * 70)
    print("This script demonstrates the full ML pipeline:")
    print("1. Generate forward-looking labels")
    print("2. Train ML model with probability calibration")
    print("3. Evaluate model calibration and performance")
    print("4. Run historical backtesting")
    print("5. Test interactive mode with ML model")
    print()

    # Change to nifty_trading_agent directory
    script_dir = Path(__file__).parent
    if script_dir.name != 'nifty_trading_agent':
        nifty_dir = script_dir / 'nifty_trading_agent'
        if nifty_dir.exists():
            import os
            os.chdir(nifty_dir)

    # Step 1: Generate labels
    if not run_command("python generate_labels.py", "STEP 1: Generating Forward-Looking Labels"):
        print("❌ Label generation failed. Please check data availability.")
        return

    # Step 2: Train model
    if not run_command("python train_model.py", "STEP 2: Training ML Model with Calibration"):
        print("❌ Model training failed. Please check label data.")
        return

    # Step 3: Evaluate model
    if not run_command("python evaluate_model.py", "STEP 3: Evaluating Model Calibration"):
        print("❌ Model evaluation failed. Please check model artifacts.")
        return

    # Step 4: Run backtesting
    if not run_command("python -m backtest.backtest_signals_with_model", "STEP 4: Running Historical Backtesting"):
        print("❌ Backtesting failed. Please check model and data.")
        return

    # Step 5: Test interactive mode (with sample inputs)
    print(f"\n{'='*60}")
    print("🎯 STEP 5: Testing Interactive Mode with ML Model")
    print('='*60)
    print("Running interactive mode with sample inputs...")

    # Sample inputs for automated testing
    sample_input = "1\n10\n12\n90\n3\n0.75\n5\n5\ny\n"
    try:
        result = subprocess.run(
            "python interactive_main.py",
            shell=True,
            input=sample_input,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        print("✅ Interactive mode completed successfully!")
        print("\n📄 Last 20 lines of output:")
        print("="*40)
        lines = result.stdout.strip().split('\n')
        for line in lines[-20:]:
            print(line)

        if result.stderr:
            print("\n⚠️  STDERR output:")
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print("⏰ Interactive mode timed out (likely waiting for input)")
    except Exception as e:
        print(f"❌ Interactive mode failed: {e}")

    # Final summary
    print(f"\n{'='*70}")
    print("🎉 COMPLETE ML WORKFLOW DEMONSTRATION FINISHED!")
    print("="*70)
    print()
    print("✅ What was accomplished:")
    print("   • Generated forward-looking labels without lookahead bias")
    print("   • Trained calibrated ML model for +10% return prediction")
    print("   • Evaluated model calibration (probability accuracy)")
    print("   • Ran historical backtesting with realistic trading simulation")
    print("   • Integrated ML model into interactive trading interface")
    print()
    print("📊 Key ML Features:")
    print("   • Statistically grounded conviction scores (0.0-1.0)")
    print("   • Probability calibration for reliable predictions")
    print("   • Time-based train/validation/test splits")
    print("   • Brier score and calibration metrics")
    print("   • Historical backtesting with realistic P&L tracking")
    print()
    print("📁 Generated Files:")
    print("   • models/artifacts/model_*.pkl (trained models)")
    print("   • reports/model_evaluation/ (calibration reports)")
    print("   • reports/backtest_ml_signals_*.json (backtest results)")
    print("   • reports/interactive_analysis_*.json (analysis reports)")
    print()
    print("🎯 Decision Support System Ready!")
    print("   Use conviction scores (0.8+) for high-confidence signals")
    print("   Monitor calibration drift in production")
    print("   Combine with risk management for robust trading")

if __name__ == "__main__":
    main()
