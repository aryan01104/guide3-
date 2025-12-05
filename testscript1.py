import requests

url = "https://api.morphik.ai/documents/"

payload = {
    "document_filters": {},
    "skip": 0,
    "limit": 10000
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print(response.json())
