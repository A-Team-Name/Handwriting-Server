from .model import Model
from ..output import Output

from PIL import Image
import onnxruntime
import numpy as np
import json

class OnnxModel(Model):
    def __init__(self, model_path: str, top_preds: int = 3):
        self.model = onnxruntime.InferenceSession(model_path)
        self.top_preds = top_preds
    
    def predict(self, img: Image.Image) -> Output:
        np_img = np.array(img)
        inputs = self.model.get_inputs()
        outputs = self.model.get_outputs()
        
        input_name = inputs[0].name
        output_names = [output.name for output in outputs]
        
        logits, softmax, softmax_ordered = self.model.run(
            output_names,
            {input_name: img}
        )
        
        top_preds = softmax_ordered[:, :, :self.top_preds, 2][0]
        top_pred_probs = softmax_ordered[:, :, :self.top_preds, 1][0]
        
        print(top_preds, flush=True)
        
        top_chars = [
            [chr(int(pred)) for pred in top_pred]
            for top_pred in top_preds
        ]
        
        return Output(
            top_chars,
            top_pred_probs.tolist()
        )