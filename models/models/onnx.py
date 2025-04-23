from .model import Model
from ..output import Output

import numpy as np
import numpy.typing as npt
import onnxruntime
import json

class OnnxModel(Model):
    """
    Model class for ONNX models. These models use .onnx binary files for inference.

    Inherits:
        Model: The base class for all models
    """
    def __init__(self, model_path: str, top_preds: int = 3):
        """Takes extra arguments for the ONNX model path and the number of top predictions to return
        Initialize the ONNX model.
        
        All models that inherit from this class should pass in the path to the ONNX model file.

        Args:
            model_path (str): The path to the ONNX model file
            top_preds (int, optional): Number of character predictions to give per character. Defaults to 3.
        """
        self.model = onnxruntime.InferenceSession(model_path)
        self.top_preds = top_preds
        super().__init__()
    
    def predict(self, img: npt.NDArray[np.ubyte]) -> Output:
        """
        Perform inference on the given image

        Args:
            img (npt.NDArray[np.ubyte]): The image to perform inference on. Image MUST have only 1 channel (i.e. 2d numpy array)

        Returns:
            Output: Output of the ONNX model
        """
        np_img = img.astype(np.float32)
        
        # Need to reshape the image to (1, 1, H, W)
        # This is because the model expects a batch size of 1 and 1 channel
        np_img = np_img.reshape(1, 1, *np_img.shape)
        
        inputs = self.model.get_inputs()
        outputs = self.model.get_outputs()
        
        input_name = inputs[0].name
        output_names = [output.name for output in outputs]
        
        softmax_ordered = self.model.run(
            output_names,
            {input_name: np_img}
        )
        
        top_preds = softmax_ordered[0][0, :self.top_preds]
        top_pred_probs = softmax_ordered[1][0, :self.top_preds]
        
        if len(top_preds.shape) == 1:
            top_preds = top_preds.reshape(1, -1)
            top_pred_probs = top_pred_probs.reshape(1, -1)
        
        top_chars = [
            [chr(int(pred)) for pred in top_pred]
            for top_pred in top_preds
        ]
        
        return Output(
            top_chars,
            top_pred_probs.tolist()
        )
