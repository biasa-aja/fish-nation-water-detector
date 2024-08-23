import requests
import random
import string

# Generate a random string of a given length
def generate_random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

# Unsplash API request
access_key = 'VTVyRGbbPNBLM5_goaq67F02lyHYOXjD9hsRbnS2qg8'
query = 'plant'
url = f'https://api.unsplash.com/search/photos?query={query}&client_id={access_key}&per_page=30'  # Increase per_page for more images

response = requests.get(url)
data = response.json()

# Download images
for photo in data['results']:
    img_url = photo['urls']['small']
    img_data = requests.get(img_url).content
    
    # Use a random string for the filename
    random_filename = generate_random_string()
    with open(f'./dataset/non-water/non_water_photo_{random_filename}.jpg', 'wb') as handler:
        handler.write(img_data)
