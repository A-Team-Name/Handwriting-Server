from .onnx import OnnxInferer

from PIL import Image
import onnxruntime
import numpy as np
import os

class HelloWorldInferer(OnnxInferer):
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'onnx/hello_world.onnx')
        super().__init__(model_path)
    
    def process_image(self, img: Image.Image) -> str:
        new_img = np.random.random(
            (
                1,  # batch: stack as many images as you like here
                1,  # channels: needs to be 1 (grayscale), pixels are 1.0 or 0.0
                1,  # height: fixed to 1 for now
                1   # width: fixed to 1 for now
            )
        ).astype(np.float32)
        
        return super().process_image(new_img)