from .onnx import OnnxModel
from ..output import Output

from PIL import Image
import onnxruntime
import numpy as np
import os

class HelloWorldModel(OnnxModel):
    """
    Hello World Model. Simply outputs "hello.world" for all inputs

    Inherits:
        OnnxModel: OnnxModel class for running ONNX models
    """
    def __init__(self):
        """
        Initialize the Hello World Model by specifying the path to the ONNX model file
        """
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'onnx/hello_world_dyn.onnx')
        super().__init__(model_path)