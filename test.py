import requests
import json

url = 'http://localhost:5000/translate'
image_path = 'hand-lambda.png'

with open(image_path, 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post(url, files=files)
    print(response.json())
