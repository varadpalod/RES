from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import random
import sys
import os

# Configure Chrome
options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")

output_file = "scrape/magicbricks_hinjewadi_extended.csv"

def save_data(cards, locality_name):
    """Extract and save data from cards element list"""
    if not cards:
        return 0
        
    data_list = []
    print(f"Extracting data from {len(cards)} cards...")
    
    for idx, card in enumerate(cards):
        try:
            # Title & Link
            try:
                title_el = card.find_element(By.CLASS_NAME, "mb-srp__card--title")
                title = title_el.text
                link = title_el.get_attribute('href')
            except:
                continue # Skip if no title/link

            # Price
            try:
                price = card.find_element(By.CLASS_NAME, "mb-srp__card__price--amount").text
            except:
                price = "N/A"
                
            # Society
            try:
                society = card.find_element(By.CLASS_NAME, "mb-srp__card__developer--name").text
            except:
                society = "N/A"

            # Area/Possession
            carpet_area = "N/A"
            possession = "N/A"
            
            summary_items = card.find_elements(By.CLASS_NAME, "mb-srp__card__summary--item")
            for item in summary_items:
                try:
                    txt = item.text.lower()
                    val = item.find_element(By.CLASS_NAME, "mb-srp__card__summary--value").text
                    if "carpet" in txt or "super area" in txt:
                        carpet_area = val
                    elif "status" in txt or "poss" in txt:
                        possession = val
                except:
                    continue

            data_list.append({
                "Title": title,
                "Price": price,
                "Society": society,
                "Area": carpet_area,
                "Possession": possession,
                "Link": link,
                "Locality": locality_name
            })
        except Exception as e:
            print(f"Skipping card due to error: {e}") 
            continue

    if data_list:
        df = pd.DataFrame(data_list)
        df.drop_duplicates(subset=['Link'], inplace=True)
        
        # Append to existing if file exists
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                df = pd.concat([existing_df, df], ignore_index=True)
                df.drop_duplicates(subset=['Link'], inplace=True)
            except:
                pass
        
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"💾 Checkpoint: Saved {len(df)} records (Total) to {output_file}")
        return len(df)
    return 0

print("Starting MagicBricks Scraper for Hinjewadi (Extended)...")

locality = "Hinjewadi"

try:
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # Hinjewadi with 2-3 BHK filter
    url = f"https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=2,3&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa&Locality={locality}&cityName=Pune"
    driver.get(url)
    
    wait = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "mb-srp__card")))
    print("Page loaded.")

    MAX_SCROLLS = 80  # Increased to get more results
    SCROLL_PAUSE_TIME = 2.0
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    for i in range(MAX_SCROLLS):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME + random.random())
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        # Checkpoint every 5 scrolls
        if (i + 1) % 5 == 0:
            cards = driver.find_elements(By.CLASS_NAME, "mb-srp__card")
            save_data(cards, locality)
            print(f"Scroll {i+1}/{MAX_SCROLLS} complete.")
            
        if new_height == last_height:
            # Try nudge
            driver.execute_script("window.scrollBy(0, -500);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            if driver.execute_script("return document.body.scrollHeight") == last_height:
                print("End of results.")
                break
        last_height = new_height

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user. Saving current data...")
except Exception as e:
    print(f"\n❌ Error: {e}")
finally:
    # Final Save
    try:
        if 'driver' in locals():
            cards = driver.find_elements(By.CLASS_NAME, "mb-srp__card")
            save_data(cards, locality)
            driver.quit()
    except:
        pass
    print("Done.")
