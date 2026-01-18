from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import pandas as pd
import time
import os

# Simple approach - just get title, price, area, link
output_file = "scrape/magicbricks_hinjewadi_new.csv"

options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")

print("🔍 Starting simple Hinjewadi scraper...")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

try:
    # Direct URL for Hinjewadi
    url = "https://www.magicbricks.com/property-for-sale/residential-real-estate?proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa&cityName=Pune&Locality=Hinjewadi"
    driver.get(url)
    
    # Wait for cards to load
    wait = WebDriverWait(driver, 15)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".mb-srp__card")))
    print("✓ Page loaded")
    
    # Scroll and collect
    data = []
    last_count = 0
    no_change_count = 0
    
    for scroll in range(30):
        # Get all cards
        cards = driver.find_elements(By.CSS_SELECTOR, ".mb-srp__card")
        
        for card in cards:
            try:
                # Title & Link
                title_elem = card.find_element(By.CSS_SELECTOR, ".mb-srp__card--title")
                title = title_elem.text.strip()
                link = title_elem.get_attribute("href")
                
                # Price
                try:
                    price = card.find_element(By.CSS_SELECTOR, ".mb-srp__card__price--amount").text.strip()
                except:
                    price = "N/A"
                
                # Area
                try:
                    area_elem = card.find_element(By.CSS_SELECTOR, ".mb-srp__card__summary--value")
                    area = area_elem.text.strip()
                except:
                    area = "N/A"
                
                # Check if already exists
                if link and not any(d['Link'] == link for d in data):
                    data.append({
                        'Title': title,
                        'Price': price,
                        'Area': area,
                        'Link': link
                    })
            except:
                continue
        
        current_count = len(data)
        print(f"Scroll {scroll + 1}/30: {current_count} properties")
        
        # Check if we're stuck
        if current_count == last_count:
            no_change_count += 1
            if no_change_count >= 3:
                print("No new properties found, stopping...")
                break
        else:
            no_change_count = 0
        
        last_count = current_count
        
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Save checkpoint every 5 scrolls
        if (scroll + 1) % 5 == 0 and data:
            df_temp = pd.DataFrame(data)
            df_temp.to_csv(output_file, index=False)
            print(f"💾 Saved {len(data)} properties")
    
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    # Final save
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"\n✅ DONE! Saved {len(df)} properties to {output_file}")
    else:
        print("\n⚠️ No data collected")
    
    driver.quit()
