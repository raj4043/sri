from flask import Flask, jsonify
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

@app.route('/scrape_data', methods=['GET'])
def scrape_data():
    # URL to scrape data from
    url = 'https://example.com'

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if request was successful
    if response.status_code == 200:
        # Parse the HTML content of the response using Beautiful Soup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract relevant data from the HTML (example: scraping titles of articles)
        articles = soup.find_all('h2', class_='article-title')
        scraped_data = [article.text for article in articles]

        return jsonify({'scraped_data': scraped_data})
    else:
        return jsonify({'error': 'Failed to fetch data from the website'}), 500

if __name__ == '__main__':
    app.run(debug=True)
