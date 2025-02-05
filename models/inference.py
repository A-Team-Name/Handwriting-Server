from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from .models import Model
from .preprocessors import Preprocessor
from .output import Output

class Inferer:
    def __init__(self, model: Model, preprocessor: Preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def process_image(self, img: Image.Image) -> Output:
        inputs: list[Image.Image] = self.preprocessor.preprocess(img)
        output_preds: list[list[str]] = []
        output_probs: list[list[float]] = []
        for image in inputs:
            output = self.model.predict(image)
            output_preds.append(output.top_preds)
            output_probs.append(output.top_probs)
            
        return Output(
            output_preds,
            output_probs
        )
