import os
import requests
from bs4 import BeautifulSoup
import time
import random

def scrape_disaster_images(query="Bengaluru flood", num_images=5, output_dir="datasets/scraped"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting web scraper for query: '{query}'")
    
    search_url = f"https://en.wikipedia.org/w/index.php?search={query.replace(' ', '+')}&title=Special:MediaSearch&type=image"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    scraped_metadata = []
    
    try:
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        img_tags = soup.find_all('img')
        
        count = 0
        for img in img_tags:
            if count >= num_images:
                break
            
            img_url = img.get('src')
            if not img_url:
                continue
                
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif img_url.startswith('/'):
                img_url = 'https://en.wikipedia.org' + img_url
                
            try:
                img_data = requests.get(img_url, headers=headers).content
                filename = f"scraped_{count}.jpg"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(img_data)
                
                # Simulate EXIF GPS extraction
                lat = 12.9716 + random.uniform(-0.1, 0.1)
                lon = 77.5946 + random.uniform(-0.1, 0.1)
                
                scraped_metadata.append({
                    'filename': filename,
                    'source_url': img_url,
                    'latitude': lat,
                    'longitude': lon
                })
                print(f"Downloaded: {filename}")
                count += 1
            except Exception as e:
                print(f"Error downloading {img_url}: {e}")
                
    except Exception as e:
        print(f"Error scraping: {e}")
        
    if len(scraped_metadata) == 0:
        print("No images found. Using placeholders.")
        for i in range(num_images):
            url = f"https://picsum.photos/seed/{random.randint(1, 1000)}/400/300"
            img_data = requests.get(url).content
            filename = f"scraped_fallback_{i}.jpg"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(img_data)
            scraped_metadata.append({
                'filename': filename,
                'source_url': url,
                'latitude': 12.9716 + random.uniform(-0.1, 0.1),
                'longitude': 77.5946 + random.uniform(-0.1, 0.1)
            })
            
    return scraped_metadata

if __name__ == "__main__":
    scrape_disaster_images()
