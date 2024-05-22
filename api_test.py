import requests
from config import *

# Test the upload endpoint
upload_url = "http://127.0.0.1:5000/upload"
files = {'files': open(TEST_FILE, 'rb')}
response = requests.post(upload_url, files=files)
print("Upload Response:", response.json())

# Test the search endpoint
search_url = "http://127.0.0.1:5000/search"
query = {"query": SEARCH_QUERY}
response = requests.post(search_url, json=query)
print("Search Response:", response.json())