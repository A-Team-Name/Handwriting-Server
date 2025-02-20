from .onnx import OnnxModel
from ..output import Output

from PIL import Image
import onnxruntime
import numpy as np
import numpy.typing as npt
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
        model_path = os.path.join(current_dir, 'onnx/lambda_calculus_char.onnx')
        super().__init__(model_path)
    
    
    def predict(self, img: npt.NDArray[np.ubyte]) -> Output:
        np_img = img.astype(np.float32)
        
        # Need to reshape the image to (1, 1, H, W)
        np_img = np_img.reshape(1, 1, *np_img.shape)

        # Run inference
        inputs: list[onnxruntime.NodeArg] = self.model.get_inputs()
        outputs: list[onnxruntime.NodeArg] = self.model.get_outputs()

        input_name: str = inputs[0].name
        output_name: list[str] = outputs[0].name
        softmax_ordered: np.ndarray
        
        Image.fromarray(np_img[0][0]).show()

        softmax_ordered = self.model.run(
            [output_name], 
            {input_name: np_img}
        )[0]
        
        softmax_ordered = softmax_ordered.reshape(1, *softmax_ordered.shape)
        
        probs: np.ndarray = softmax_ordered[:, :self.top_preds, 1]
        chars: np.ndarray = softmax_ordered[:, :self.top_preds, 0]
        
        top_chars = [
            [chr(int(pred)) for pred in top_pred]
            for top_pred in chars
        ]
        
        top_probs = [
            [float(prob) for prob in top_prob]
            for top_prob in probs
        ]
        

        
        return Output(
            top_chars,
            top_probs
        )
    