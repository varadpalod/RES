from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import sys

# Configure Chrome Options
options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")
# options.add_argument("--headless") # Uncomment to run invisibly

print("Starting scraper...")
try:
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
except Exception as e:
    print(f"Error initializing driver: {e}")
    sys.exit(1)

url = "https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=2,3&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa&Locality=Kharadi&cityName=Pune"

print(f"Navigating to {url}...")
driver.get(url)

wait = WebDriverWait(driver, 25)

# Wait for cards to load
try:
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "mb-srp__card")))
    print("Page loaded.")
except:
    print("Timeout waiting for content.")
    driver.quit()
    sys.exit(1)

# Scroll to load more cards
print("Scrolling to load more properties...")
for i in range(5):
    driver.execute_script("window.scrollBy(0, 1500);")
    time.sleep(2)
    print(f"Scroll {i+1}/5 completed")

# Find all cards
cards = driver.find_elements(By.CLASS_NAME, "mb-srp__card")
print(f"Found {len(cards)} property cards. Parsing data...")

data_list = []

for idx, card in enumerate(cards):
    try:
        # Title & Link
        try:
            title_el = card.find_element(By.CLASS_NAME, "mb-srp__card--title")
            title = title_el.text
            link = title_el.get_attribute('href')
        except:
            title = "N/A"
            link = "N/A"
        
        # Price
        try:
            price = card.find_element(By.CLASS_NAME, "mb-srp__card__price--amount").text
        except:
            price = "N/A"
            
        # Society / Developer
        try:
            society = card.find_element(By.CLASS_NAME, "mb-srp__card__developer--name").text
        except:
            society = "N/A"
            
        # Summary Details (Carpet, Status, Floor etc)
        carpet_area = "N/A"
        possession = "N/A"
        
        summary_items = card.find_elements(By.CLASS_NAME, "mb-srp__card__summary--item")
        for item in summary_items:
            try:
                label_el = item.find_element(By.CLASS_NAME, "mb-srp__card__summary--label")
                value_el = item.find_element(By.CLASS_NAME, "mb-srp__card__summary--value")
                label = label_el.text.lower()
                val = value_el.text
                
                if "carpet" in label or "super area" in label:
                    carpet_area = val
                elif "status" in label or "poss" in label:
                    possession = val
            except:
                continue
                
        data_list.append({
            "Title": title,
            "Price": price,
            "Society": society,
            "Area": carpet_area,
            "Possession": possession,
            "Link": link
        })
        
    except Exception as e:
        print(f"Error parsing card {idx}: {e}")
        continue

driver.quit()

# Save structured CSV
output_file = "magicbricks_kharadi_structured.csv"
if data_list:
    df = pd.DataFrame(data_list)
    df.to_csv(output_file, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
    print(f"\n✅ Scraped {len(df)} properties.")
    print(f"✅ Data saved to: {output_file}")
    print("\nSample Data:")
    print(df[['Title', 'Price', 'Area']].head())
else:
    print("No data extracted.")
