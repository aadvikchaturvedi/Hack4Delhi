import requests
import pandas as pd

class IndiaDataIngestor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.data.gov.in/resource/"

    def fetch_resource(self, resource_id, limit=1000):
        """Fetches central or state procurement data."""
        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit
        }
        response = requests.get(f"{self.base_url}{resource_id}", params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            return pd.DataFrame(records)
        raise Exception(f"API Error: {response.status_code}")

# Example: Resource ID for Assam Public Procurement is often found on data.gov.in