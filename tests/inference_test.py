import pytest
from models.inference import Inferer
from models.output import Output

# create a mock for a preprocessor and model

from unittest.mock import MagicMock, patch
from io import BytesIO

import numpy as np


def mock_preprocessor():
    """Mock preprocessor"""
    preprocessor = MagicMock()
    preprocessor.preprocess.return_value = [np.ones((128, 128), dtype=np.ubyte) * 255] * 5
    return preprocessor

def mock_model():
    """Mock model"""
    model = MagicMock()
    model.predict.return_value = Output(**{
        "top_preds": [["λ", "x", "."]],
        "top_probs": [[0.9, 0.8, 0.7]]
    })
    return model

def test_model_concat():
    """Test to see if model concat works"""
    
    preprocessor = mock_preprocessor()
    model = mock_model()
    inferer = Inferer(model, preprocessor)

    img = np.ones((128, 128), dtype=np.ubyte) * 255
    output = inferer.process_image(img)

    assert len(output.top_preds) == 5, "Output preds length mismatch"
    assert len(output.top_probs) == 5, "Output probs length mismatch"

    for preds in output.top_preds:
        assert isinstance(preds, list), "Preds should be a list"
        assert len(preds) == 3, "Preds should have 3 elements"

    for probs in output.top_probs:
        assert isinstance(probs, list), "Probs should be a list"
        assert len(probs) == 3, "Probs should have 3 elements"
        
        
def test_space():
    """Test to see if adding spaces works"""
    pass

def test_tabs():
    """Test to see if adding tabs works"""
    pass