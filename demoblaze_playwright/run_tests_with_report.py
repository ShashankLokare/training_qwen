#!/usr/bin/env python3
"""
Script to run Playwright tests with Allure reporting.
Automatically opens the Allure report after test execution.
"""
import subprocess
import sys
import os
import shutil
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
ALLURE_BIN = SCRIPT_DIR / "allure-commandline" / "allure-2.25.0" / "bin" / "allure"
ALLURE_RESULTS = SCRIPT_DIR / "allure-results"
ALLURE_REPORT = SCRIPT_DIR / "allure-report"

def main():
    print("=" * 80)
    print("Running Playwright Tests with Allure Reporting")
    print("=" * 80)
    
    # Clean up previous results and reports
    if ALLURE_RESULTS.exists():
        print(f"\n🧹 Cleaning previous results from {ALLURE_RESULTS}")
        shutil.rmtree(ALLURE_RESULTS)
    
    if ALLURE_REPORT.exists():
        print(f"🧹 Cleaning previous report from {ALLURE_REPORT}")
        shutil.rmtree(ALLURE_REPORT)
    
    # Run pytest with allure
    print("\n🧪 Running tests...")
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--headed",
        "--alluredir=allure-results"
    ]
    
    result = subprocess.run(pytest_cmd, cwd=SCRIPT_DIR)
    
    print("\n" + "=" * 80)
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the report for details.")
    print("=" * 80)
    
    # Generate and open Allure report
    if ALLURE_RESULTS.exists() and any(ALLURE_RESULTS.iterdir()):
        print(f"\n📊 Generating Allure report...")
        
        # Generate report
        generate_cmd = [
            str(ALLURE_BIN),
            "generate",
            str(ALLURE_RESULTS),
            "-o",
            str(ALLURE_REPORT),
            "--clean"
        ]
        
        subprocess.run(generate_cmd, check=True)
        
        print(f"✅ Report generated at: {ALLURE_REPORT}")
        
        # Open report
        print("\n🌐 Opening Allure report in browser...")
        open_cmd = [
            str(ALLURE_BIN),
            "open",
            str(ALLURE_REPORT)
        ]
        
        # This will keep the server running and block until Ctrl+C
        try:
            subprocess.run(open_cmd)
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down report server...")
    else:
        print("\n⚠️  No test results found. Report not generated.")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
