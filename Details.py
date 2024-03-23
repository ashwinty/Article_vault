import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to scrape the articles from all <div class="row box-post"> elements on a single page
def scrape_articles(url):
    articles_data = []

    # Send a GET request to the URL
    response = requests.get(url)
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all div elements with class 'row box-post'
    articles = soup.find_all('div', class_='row box-post')

    for article in articles:
        # Extract article info
        article_info = article.find('div', class_='box-post-info')

        # Extract URLs from all <a> tags within the box-post-info div
        url_elems = article_info.find_all('a')
        
        # Check each URL
        for url_elem in url_elems:
            url = url_elem['href']
            # Check if URL matches the desired format within the given tag
            if url and '/20' in url and '/author/' not in url and '/journal-topic/' not in url:
                # Extract other information
                title_elem = article_info.find('h2')
                title = title_elem.text.strip() if title_elem else None

                text_elem = article_info.find('div', class_='category-description')
                text = text_elem.text.strip() if text_elem else None

                tag_elem = article_info.find('span', class_='category-name')
                tag = tag_elem.text.strip() if tag_elem else None

                author_date_elem = article_info.find('div', class_='category-date')
                if author_date_elem:
                    author_elem = author_date_elem.find('a', class_='author')
                    author = author_elem.text.strip() if author_elem else None

                    date_elem = author_date_elem.find('span', class_='date')
                    date = date_elem.text.strip() if date_elem else None
                else:
                    author = None
                    date = None

                # Append the data to the list
                articles_data.append({
                    'Title': title,
                    'URL': url,  # Use the extracted URL directly
                    'Text': text,
                    'Tag': tag,
                    'Author': author,
                    'Date': date,
                })

                # Break after finding a valid URL to avoid duplicate entries
                break

    return articles_data

# Function to scrape articles from all pages
def scrape_all_articles(base_url, num_pages):
    all_articles = []
    for page_num in range(1, num_pages + 1):
        url = f"{base_url}page/{page_num}/"
        articles = scrape_articles(url)
        all_articles.extend(articles)
    return all_articles


# URL of the webpage to scrape
base_url = 'https://theobjectivestandard.com/archive/'

# Specify the number of pages to scrape
num_pages = 184

# Scrape the articles from all pages
articles = scrape_all_articles(base_url, num_pages)

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(articles)

# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file
df.to_csv('articles_data2.csv', index=False)












# import requests
# from bs4 import BeautifulSoup
# import pandas as pd

# # Function to scrape the articles from all <div class="row box-post"> elements on a single page
# def scrape_articles(url):
#     articles_data = []

#     # Send a GET request to the URL
#     response = requests.get(url)
#     # Parse the HTML content
#     soup = BeautifulSoup(response.content, 'html.parser')

#     # Find all div elements with class 'row box-post'
#     articles = soup.find_all('div', class_='row box-post')
#     # print(articles)

#     for article in articles:
#         # Extract article info
#         article_info = article.find('div', class_='box-post-info')

#         # Extract title
#         title_elem = article_info.find('h2')
#         title = title_elem.text.strip() if title_elem else None

#         # Extract URL from the second <a> tag
#         url_elem = article_info.find_all('a')[2]['href'] if article_info.find_all('a') else None

#         # Extract text
#         text_elem = article_info.find('div', class_='category-description')
#         text = text_elem.text.strip() if text_elem else None

#         # Extract tag
#         tag_elem = article_info.find('span', class_='category-name')
#         tag = tag_elem.text.strip() if tag_elem else None

#         # Extract author and date
#         author_date_elem = article_info.find('div', class_='category-date')
#         if author_date_elem:
#             author_elem = author_date_elem.find('a', class_='author')
#             author = author_elem.text.strip() if author_elem else None

#             date_elem = author_date_elem.find('span', class_='date')
#             date = date_elem.text.strip() if date_elem else None
#         else:
#             author = None
#             date = None

#         # Append the data to the list
#         articles_data.append({
#             'Title': title,
#             'URL': url_elem,  # Use the extracted URL directly
#             'Text': text,
#             'Tag': tag,
#             'Author': author,
#             'Date': date,
#         })

#     return articles_data

# # Function to scrape articles from all pages
# def scrape_all_articles(base_url, num_pages):
#     all_articles = []
#     for page_num in range(1, num_pages + 1):
#         url = f"{base_url}page/{page_num}/"
#         articles = scrape_articles(url)
#         all_articles.extend(articles)
#     return all_articles


# # URL of the webpage to scrape
# base_url = 'https://theobjectivestandard.com/archive/'

# # Specify the number of pages to scrape
# num_pages = 184

# # Scrape the articles from all pages
# articles = scrape_all_articles(base_url, num_pages)

# # Convert the list of dictionaries into a DataFrame
# df = pd.DataFrame(articles)

# # Display the DataFrame
# print(df)

# # Save the DataFrame to a CSV file
# df.to_csv('articles_data.csv', index=False)














