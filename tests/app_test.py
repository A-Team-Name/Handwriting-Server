import pytest
from app import app
from io import BytesIO
import json

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client
        
def test_test_gpu(client):
    """Test the /test endpoint"""
    response = client.get("/test")
    assert response.status_code == 200
    
def test_translate(client):
    """Test the /translate endpoint"""
    # Mock the request data
    
    data = {
        'model': 'cnn',
    }
    
    with open("tests/test_img.png", "rb") as image_file:
        files = {
            'image': (image_file, "test_img.png"),  # File object and filename
            'json': (BytesIO(json.dumps(data).encode('utf-8')), "data.json")  # JSON data as BytesIO
        }
    
        response = client.post("/translate", data=files, content_type='multipart/form-data')
    
    json_data = json.loads(response.data)
    
    assert "top_preds" in json_data
    assert "top_probs" in json_data

    assert all([c in json_data["top_preds"][i] for i, c in enumerate(["λ", "x", ".", "x", "y"])])
    
    assert response.status_code == 200
    assert b"top_preds" in response.data
    assert b"top_probs" in response.data