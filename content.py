import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extracting main image URL
            main_image_url = ''
            main_image_element = soup.find('div', class_='box-content-img')
            if main_image_element:
                img_tag = main_image_element.find('img')
                # print(img_tag)
                if img_tag and 'src' in img_tag.attrs:
                    main_image_url = img_tag['data-src']
            
            # Extracting article text
            article_text = ''
            article_content = soup.find('div', class_='box-main-post-content')
            if article_content:
                paragraphs = article_content.find_all('p')
                article_text = ''.join(p.text.strip() for p in paragraphs)
            
            return main_image_url, article_text
        else:
            print(f"Failed to fetch {url}. Status code: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")
        return None, None

# Read the CSV file containing website URLs
df = pd.read_csv('urls.csv')

# Create empty lists to store extracted data
main_image_urls = []
article_texts = []

# Iterate through each URL
for url in df['Website']:
    main_image_url, article_text = scrape_website(url)
    main_image_urls.append(main_image_url)
    article_texts.append(article_text)

# Add the extracted data to the DataFrame
df['Main Image URL'] = main_image_urls
df['Article Text'] = article_texts

# Save the DataFrame to a new JSON file
df.to_json('extracted_data.json', orient='records')

print("Extraction completed and saved to 'extracted_data.json'.")
