import requests
import json

with open('test.png', 'rb') as image_file:
    print(requests
        .post(
            'http://localhost:5000/translate',
            files = {
                'image': image_file,
                'json': json.dumps({ 'model': 'shape-lambda-calculus' }),
            },
        )
        # .json()
        .text
    )
