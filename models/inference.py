from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from .models import Model
from .preprocessors import Preprocessor
from .output import Output

class Inferer:
    """
    Class to perform inference on a model with a given preprocessor
    
    Args:
        model (Model): The model to perform inference on
        preprocessor (Preprocessor): The preprocessor to use to preprocess the image
    """
    def __init__(self, model: Model, preprocessor: Preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def process_image(self, img: Image.Image) -> Output:
        """
        Perform inference on the given image

        Args:
            img (Image.Image): The image to perform inference on

        Returns:
            Output: The output of the model
        """
        inputs: list[Image.Image] = self.preprocessor.preprocess(img)
        output_preds: list[list[str]] = []
        output_probs: list[list[float]] = []
        for image in inputs:
            output = self.model.predict(image)
            output_preds += output.top_preds
            output_probs += output.top_probs

        return Output(
            output_preds,
            output_probs
        )
