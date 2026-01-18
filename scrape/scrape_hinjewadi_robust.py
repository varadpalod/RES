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

# Configure Chrome - more stable settings
options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")

output_file = "scrape/magicbricks_hinjewadi_extended.csv"
locality = "Hinjewadi"

def extract_card_data(card, locality_name):
    """Extract data from a single card with error handling"""
    try:
        # Title & Link
        title_el = card.find_element(By.CLASS_NAME, "mb-srp__card--title")
        title = title_el.text
        link = title_el.get_attribute('href')
        
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
        
        return {
            "Title": title,
            "Price": price,
            "Society": society,
            "Area": carpet_area,
            "Possession": possession,
            "Link": link,
            "Locality": locality_name
        }
    except:
        return None

def save_data(data_list):
    """Save data to CSV"""
    if not data_list:
        return 0
    
    df = pd.DataFrame(data_list)
    df.drop_duplicates(subset=['Link'], inplace=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    return len(df)

print(f"Starting Robust MagicBricks Scraper for {locality}...")
print(f"Output: {output_file}")

driver = None
data_list = []

try:
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    url = f"https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=2,3&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa&Locality={locality}&cityName=Pune"
    driver.get(url)
    
    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "mb-srp__card")))
    print("✓ Page loaded")
    
    # Reduced scrolls to avoid crashes
    MAX_SCROLLS = 20
    SCROLL_PAUSE = 2.5
    
    for i in range(MAX_SCROLLS):
        try:
            # Get cards BEFORE scrolling to avoid lost data
            cards = driver.find_elements(By.CLASS_NAME, "mb-srp__card")
            
            # Extract data from new cards
            for card in cards:
                card_data = extract_card_data(card, locality)
                if card_data and card_data not in data_list:
                    # Check if link already exists
                    existing_links = [d['Link'] for d in data_list]
                    if card_data['Link'] not in existing_links:
                        data_list.append(card_data)
            
            print(f"Scroll {i+1}/{MAX_SCROLLS}: Collected {len(data_list)} unique properties")
            
            # Save checkpoint every 3 scrolls
            if (i + 1) % 3 == 0:
                saved = save_data(data_list)
                print(f"💾 Checkpoint saved: {saved} properties")
            
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE + random.random())
            
        except Exception as e:
            print(f"Error on scroll {i+1}: {e}")
            # Still try to save what we have
            save_data(data_list)
            break
    
except KeyboardInterrupt:
    print("\n🛑 Interrupted by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
finally:
    # ALWAYS try to save data
    if data_list:
        final_count = save_data(data_list)
        print(f"\n✓ Final save: {final_count} properties saved to {output_file}")
    else:
        print("\n⚠ No data collected")
    
    # Close browser
    if driver:
        try:
            driver.quit()
            print("✓ Browser closed")
        except:
            pass

print("Done!")
