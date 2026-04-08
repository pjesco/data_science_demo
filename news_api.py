import requests
from dotenv import load_dotenv
import os

load_dotenv()

my_api_key = os.getenv('NEWS_API')

url = f'https://newsapi.org/v2/everything?q=bitcoin&apiKey={my_api_key}'

response = requests.get(url).json()

print(response['articles'][0]['title'])
print(response['articles'][0].keys())