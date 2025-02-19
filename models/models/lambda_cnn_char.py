from .onnx import OnnxModel
from ..output import Output

from PIL import Image
import onnxruntime
import numpy as np
import os

class LambdaCNNChar(OnnxModel):
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
        model_path = os.path.join(current_dir, 'onnx/char_model_lamda_calculus.onnx')
        super().__init__(model_path)
    
    
    def predict(self, img: Image.Image) -> Output:
        
        input_image = np.asarray(img).astype(np.float32)

        # Run inference
        inputs: list[onnxruntime.NodeArg] = self.model.get_inputs()
        outputs: list[onnxruntime.NodeArg] = self.model.get_outputs()

        input_name: str = inputs[0].name
        output_name: list[str] = outputs[0].name
        softmax_ordered: np.ndarray
        

        model_outputs = self.model.run(
            [output_name], 
            {input_name: input_image}
        )

        softmax_ordered = model_outputs[0]
        probs: np.ndarray = softmax_ordered[:, 1]
        chars: np.ndarray = softmax_ordered[:, 0]

        top_3_chars: np.ndarray = chars[:3]
        top_3_probs: np.ndarray = probs[:3]
        
        return Output(
            [top_3_chars],
            [top_3_probs]
        )
    