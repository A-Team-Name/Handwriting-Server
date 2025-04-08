import pytest
from models.preprocessors import LinePreprocessor, CharPreprocessor
import numpy as np

def test_empty_char():
    """Test the CharPreprocessor with an empty image"""
    preprocessor = CharPreprocessor()
    image = np.zeros((128, 128), dtype=np.ubyte)
    result = preprocessor.preprocess(image)
    assert result == []
    
def test_empty_line():
    """Test the LinePreprocessor with an empty image"""
    preprocessor = LinePreprocessor()
    image = np.zeros((128, 128), dtype=np.ubyte)
    result = preprocessor.preprocess(image)
    assert result == []
    
def test_char_separation():
    """Test the CharPreprocessor with an image containing multiple characters"""
    preprocessor = CharPreprocessor()
    image = np.ones((128, 128), dtype=np.ubyte)*255
    # Simulate two characters in the image
    image[30:70, 30:70] = 0  # Character 1
    image[30:70, 80:120] = 0  # Character 2
    result = preprocessor.preprocess(image)
    assert len(result) == 2  # Two characters should be detected
    
def test_line_separation():
    """Test the LinePreprocessor with an image containing multiple lines"""
    preprocessor = LinePreprocessor()
    image = np.ones((128, 128), dtype=np.ubyte)*255
    # Simulate two lines in the image
    image[30:70, :] = 0  # Line 1
    image[80:120, :] = 0  # Line 2
    result = preprocessor.preprocess(image)
    assert len(result) == 2  # Two lines should be detected
    assert result[0].shape[0] == 42  # Each line should be 42 pixels tall
    assert result[1].shape[0] == 42  # Each line should be 42 pixels tall
    assert result[0].shape[1] == 128  # Each line should be 128 pixels wide
    assert result[1].shape[1] == 128  # Each line should be 128 pixels wide
    
def test_char_on_equals():
    """Test the CharPreprocessor with a sample image"""
    preprocessor = CharPreprocessor()
    image = np.ones((128, 128), dtype=np.ubyte)*255
    # Simulate an '=' character in the image
    image[30:70, 30:70] = 0  # Character 1
    image[80:120, 30:70] = 0  # Character 2
    result = preprocessor.preprocess(image)
    assert len(result) == 1  # One character should be detected
    assert result[0].shape == (92, 42)  # Character should be 92x42 pixels
    
def test_char_invalid_shape():
    """Test the CharPreprocessor with an image of invalid shape"""
    preprocessor = CharPreprocessor()
    image = np.ones((5,5,5), dtype=np.ubyte)*255
    # Simulate an invalid character in the image
    with pytest.raises(ValueError):
        preprocessor.preprocess(image)
        
def test_line_invalid_shape():
    """Test the LinePreprocessor with an image of invalid shape"""
    preprocessor = LinePreprocessor()
    image = np.ones((5,5,5), dtype=np.ubyte)*255
    # Simulate an invalid line in the image
    with pytest.raises(ValueError):
        preprocessor.preprocess(image)
