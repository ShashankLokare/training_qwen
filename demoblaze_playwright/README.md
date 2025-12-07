# DemoBlaze Playwright Test Automation

This project contains automated E2E tests for the DemoBlaze e-commerce website using Playwright and Python.

## Features

- ✅ Tests run in **headed mode with Chrome** by default
- 📊 **Allure reporting** integrated with standalone JAR
- 🚀 Automatic report generation and opening after test execution
- 📸 Screenshots captured at each test step
- 📝 Detailed test steps and annotations

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Installation

1. Install required Python packages:
```bash
pip install pytest playwright allure-pytest
```

2. Install Playwright browsers:
```bash
playwright install chromium chrome
```

## Running Tests

### Option 1: Run tests with Allure report (Recommended)

Simply execute the Python script:
```bash
cd demoblaze_playwright
python run_tests_with_report.py
```

This will:
1. Clean previous test results
2. Run all tests in headed Chrome mode
3. Generate Allure report
4. Automatically open the report in your browser
5. Keep the report server running until you press Ctrl+C

### Option 2: Run tests without report

```bash
cd demoblaze_playwright
pytest tests/ -v
```

### Option 3: Run tests with Allure manually

```bash
cd demoblaze_playwright
pytest tests/ -v --alluredir=allure-results
./allure-commandline/allure-2.25.0/bin/allure serve allure-results
```

## Project Structure

```
demoblaze_playwright/
├── allure-commandline/         # Allure standalone installation
├── tests/
│   └── test_search_and_purchase.py  # Main test file
├── allure-results/             # Test results (generated)
├── allure-report/              # HTML report (generated)
├── run_tests_with_report.py    # Main test runner script
└── README.md                   # This file
```

## Test Scenarios

### test_buy_product (Data-driven)
The test is parameterized to run against multiple products loaded from `testdata/products.xlsx`:

**Current Products:**
- iPhone 6 32GB
- Nexus 6

**Test Flow for each product:**
- Navigates to DemoBlaze website
- Selects the specified product from the product list
- Adds product to cart
- Proceeds to checkout
- Fills in order details
- Completes purchase
- Verifies success message

**Screenshots captured at each step:**
- Homepage after navigation
- Product details page
- Add to cart confirmation
- Cart page with product
- Order form
- Filled order form
- Success message

## Allure Report Features

The Allure report includes:
- Test execution summary
- Detailed test steps with descriptions
- Screenshots at each major step
- Test severity and categorization
- Execution timeline
- Test history tracking

## Configuration

### Browser Configuration
All Playwright configuration is centralized in `pytest.ini`:
- **Headed mode**: `--headed` flag in pytest addopts and explicit `--headed` in the test runner script
- **Browser**: Chromium with Chrome channel

The configuration in `pytest.ini`:
```ini
[tool:pytest]
addopts = --browser chromium --headed
```

Tests always run in headed mode by default. The `run_tests_with_report.py` script explicitly includes the `--headed` flag to ensure headed execution.

## Troubleshooting

### Chrome not found
If you get an error about Chrome not being found, install it:
```bash
playwright install chrome
```

### Allure command not working
The project uses a standalone Allure installation in `allure-commandline/`. If you encounter issues, you can download it manually from:
https://github.com/allure-framework/allure2/releases

### Port already in use
If the Allure server port is already in use, kill the process:
```bash
lsof -ti:PORT | xargs kill -9
```

## Contributing

Feel free to add more test cases in the `tests/` directory following the same pattern.

## License

This project is for educational and testing purposes.
