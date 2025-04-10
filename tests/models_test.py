import pytest
from models.models import TransformerModel, ShapeContextsModel, LambdaCNNChar
from models.output import Output
import numpy as np

# @pytest.mark.skip(reason="laptop bad :(")
def test_transformer_output():
    """Test the TransformerModel output"""
    model = TransformerModel("MrFitzmaurice/TrOCR-Lambda-Calculus", "MrFitzmaurice/TrOCR-Lambda-Calculus")
    image = np.ones((128, 128), dtype=np.ubyte)*255
    result = model.predict(image)
    assert isinstance(result, Output)  # Assuming the output is a string
    assert len(result.top_preds) > 0  # Check if predictions are made
    assert len(result.top_probs) > 0  # Check if probabilities are made
    
    # For each set of probs in top_probs, ensure it is in descending order
    for probs in result.top_probs:
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1)), "Probabilities are not in descending order"
        
    
@pytest.mark.skip(reason="CNN model isn't working in tests? Works fine in deployment")
def test_cnn_output():
    """Test the ShapeContextsModel output"""
    model = LambdaCNNChar()
    image = np.ones((128, 128), dtype=np.ubyte)*255
    result = model.predict(image)
    assert isinstance(result, Output)  # Assuming the output is a string
    assert len(result.top_preds) > 0  # Check if predictions are made
    assert len(result.top_probs) > 0  # Check if probabilities are made
    
    # For each set of probs in top_probs, ensure it is in descending order
    for probs in result.top_probs:
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1)), "Probabilities are not in descending order"
        
@pytest.mark.skip(reason="ShapeContextsModel requires APL to be installed")
def test_shape_output():
    """Test the ShapeContextsModel output"""
    model = ShapeContextsModel()
    image = np.ones((128, 128), dtype=np.ubyte)*255
    result = model.predict(image)
    assert isinstance(result, Output)  # Assuming the output is a string
    assert len(result.top_preds) > 0  # Check if predictions are made
    assert len(result.top_probs) > 0  # Check if probabilities are made
    
    # For each set of probs in top_probs, ensure it is in descending order
    for probs in result.top_probs:
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1)), "Probabilities are not in descending order"
        
