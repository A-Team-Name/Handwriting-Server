import pytest
from app import app
from io import BytesIO
import json

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client
    
def send_and_receive(client, data):
    """Send a request to the server and receive the response"""
    
    response = client.post("/translate", data=data, content_type='multipart/form-data')
    return response
        
def test_test_gpu(client):
    """Test the /test endpoint"""
    response = client.get("/test")
    assert response.status_code == 200
    
def test_translate(client):
    """Test the /translate endpoint"""
    # Mock the request data
    
    data = {
        'model': 'cnn-lambda-calculus',
    }
    
    with open("tests/test_img.png", "rb") as image_file:
        files = {
            'image': (image_file, "test_img.png"),  # File object and filename
            'json': (BytesIO(json.dumps(data).encode('utf-8')), "data.json")  # JSON data as BytesIO
        }
    
        response = send_and_receive(client, files)
    
    json_data = json.loads(response.data)
    
    assert "top_preds" in json_data
    assert "top_probs" in json_data
    assert len(json_data["top_preds"]) == len(json_data["top_probs"])
    assert all(len(pred) == len(prob) for pred, prob in zip(json_data["top_preds"], json_data["top_probs"]))

    assert all([c in json_data["top_preds"][i] for i, c in enumerate(["λ", "x", ".", "x", "y"])])
    
    assert response.status_code == 200
    
def test_no_image(client):
    """Test the /translate endpoint without an image"""
    files = {
        'json': (BytesIO(json.dumps({"model": "incorrect_model"}).encode('utf-8')), "data.json"),
    }
    
    response = send_and_receive(client, files)
    
    assert response.status_code == 400
    assert b"No image provided" in response.data
    
def test_no_model(client):
    """Test the /translate endpoint without an image"""
    with open("tests/test_img.png", "rb") as image_file:
        files = {
            'image': (image_file, "test_img.png"),  # File object and filename
        }
    
        response = send_and_receive(client, files)
    
    assert response.status_code == 400
    assert b"No model provided" in response.data
    
def test_incorrect_model(client):
    """Test the /translate endpoint with an incorrect model"""
    with open("tests/test_img.png", "rb") as image_file:
        files = {
            'image': (image_file, "test_img.png"),  # File object and filename
            'json': (BytesIO(json.dumps({"model": "incorrect_model"}).encode('utf-8')), "data.json")  # JSON data as BytesIO
        }
    
        response = send_and_receive(client, files)
    
    json_data = json.loads(response.data)
    
    assert response.status_code == 400
    assert json_data["error"] == "Model not recognized"