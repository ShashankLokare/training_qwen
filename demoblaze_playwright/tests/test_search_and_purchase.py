import pytest
import allure
import pandas as pd

# Load products from Excel
df = pd.read_excel('testdata/products.xlsx', engine='openpyxl')
products = df['Product'].tolist()

@pytest.mark.parametrize("product", products)
@allure.feature("E-commerce")
@allure.story("Product Purchase Flow")
@allure.title("Test: Buy {product} from DemoBlaze")
@allure.description("End-to-end test for purchasing {product} from demoblaze.com")
@allure.severity(allure.severity_level.CRITICAL)
def test_buy_product(page, product):
    # Adjust product name for cart display (site shows 'gb' instead of 'GB')
    product_cart = product.replace('GB', 'gb')

    with allure.step("Navigate to DemoBlaze homepage"):
        page.goto("https://demoblaze.com")
        allure.attach(page.screenshot(), name="Homepage", attachment_type=allure.attachment_type.PNG)

    with allure.step(f"Select {product} from product list"):
        page.wait_for_selector(f"text={product}", timeout=10000)
        page.click(f"text={product}")
        allure.attach(page.screenshot(), name="Product Page", attachment_type=allure.attachment_type.PNG)

    with allure.step("Add product to cart"):
        page.wait_for_selector("text=Add to cart", timeout=10000)
        page.on("dialog", lambda dialog: dialog.accept())
        page.click("text=Add to cart")
        page.wait_for_timeout(2000)
        allure.attach(page.screenshot(), name="Add to Cart Confirmation", attachment_type=allure.attachment_type.PNG)

    with allure.step("Navigate to cart"):
        page.click("text=Cart")
        page.wait_for_selector(f"text={product_cart}", timeout=10000)
        allure.attach(page.screenshot(), name="Cart Page", attachment_type=allure.attachment_type.PNG)
    
    with allure.step("Initiate checkout process"):
        page.click("button:has-text('Place Order')")
        page.wait_for_selector("#name", timeout=5000)
        allure.attach(page.screenshot(), name="Order Form", attachment_type=allure.attachment_type.PNG)
    
    with allure.step("Fill in customer and payment details"):
        page.fill("#name", "Test User")
        page.fill("#country", "USA")
        page.fill("#city", "New York")
        page.fill("#card", "1234567890123456")
        page.fill("#month", "12")
        page.fill("#year", "2025")
        allure.attach(page.screenshot(), name="Filled Order Form", attachment_type=allure.attachment_type.PNG)
    
    with allure.step("Complete purchase"):
        page.click("button:has-text('Purchase')")
        page.wait_for_selector("text=Thank you for your purchase!", timeout=10000)
        allure.attach(page.screenshot(), name="Success Message", attachment_type=allure.attachment_type.PNG)
    
    with allure.step("Close confirmation dialog"):
        page.click("button:has-text('OK')")
        allure.attach("Purchase completed successfully", name="Test Result", attachment_type=allure.attachment_type.TEXT)
