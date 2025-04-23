from .onnx import OnnxModel
from ..output import Output

from PIL import Image
import onnxruntime
import numpy as np
import numpy.typing as npt
import os

class APLCNNChar(OnnxModel):
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
        model_path = os.path.join(current_dir, 'onnx/CNN_APL.onnx')
        super().__init__(model_path)
        
    def predict(self, img: npt.NDArray[np.ubyte]):
        img = Image.fromarray(img.astype(np.uint8))
        img_resized = img.resize((60, 60))
        
        img = np.array(img_resized)

        # Pad image with 2 pixels (white)
        img_padded = 255 - np.pad(img, pad_width=2, mode='constant', constant_values=255)

        return super().predict(img_padded)
    
