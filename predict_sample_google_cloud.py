import requests


url = 'https://airbnb-price-predictor-558797510368.us-central1.run.app/predict'

# Sample listing data
listing = {
    'latitude': 40.75362,
    'longitude': -73.98377,
    'minimum_nights': 5,
    'number_of_reviews': 45,
    'reviews_per_month': 0.38,
    'calculated_host_listings_count': 2,
    'availability_365': 355,
    'neighbourhood_group': 'Manhattan',
    'room_type': 'Entire home/apt',
    'room_neighborhood': 'Entire home/apt_Manhattan'
}

# Send request
try:
    response = requests.post(url, json=listing, timeout=10)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    print(response.json())
except requests.exceptions.ConnectionError as e:
    print(f"Error: Could not connect to the server at {url}.")
    print(f"Details: {e}")
except requests.exceptions.Timeout as e:
    print(f"Error: Connection to the server at {url} timed out.")
    print(f"Details: {e}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")