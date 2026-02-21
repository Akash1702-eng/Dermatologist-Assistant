# test_api.py
import requests
import json

def test_health():
    response = requests.get('http://localhost:5000/api/health')
    print("Health Check:", response.json())

if __name__ == '__main__':
    test_health()