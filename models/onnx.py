from .inference import Inferer

from PIL import Image
import onnxruntime
import numpy as np
import json

class OnnxInferer(Inferer):
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)
    
    def process_image(self, img: Image.Image) -> str:
        np_img = np.array(img)
        inputs = self.model.get_inputs()
        outputs = self.model.get_outputs()
        
        input_name = inputs[0].name
        output_names = [output.name for output in outputs]
        
        logits, softmax, softmax_ordered = self.model.run(
            output_names,
            {input_name: img}
        )
        
        top3_preds = softmax_ordered[:, :, :3, 2]
        top3_pred_probs = softmax_ordered[:, :, :3, 1]
        
        print(softmax_ordered.shape, flush=True)
        # print(top3_pred_probs, flush=True)
        
        return json.dumps({
            "top3_preds": top3_preds.tolist(),
            "top3_pred_probs": top3_pred_probs.tolist()
        })