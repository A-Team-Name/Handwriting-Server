from .onnx import OnnxModel
from ..output import Output

from PIL import Image
import onnxruntime
import numpy as np
import os

class HelloWorldModel(OnnxModel):
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'onnx/hello_world.onnx')
        super().__init__(model_path)
    
    def predict(self, img: Image.Image) -> Output:
        new_img = np.random.random(
            (
                1,  # batch: stack as many images as you like here
                1,  # channels: needs to be 1 (grayscale), pixels are 1.0 or 0.0
                1,  # height: fixed to 1 for now
                1   # width: fixed to 1 for now
            )
        ).astype(np.float32)
        
        return super().predict(new_img)