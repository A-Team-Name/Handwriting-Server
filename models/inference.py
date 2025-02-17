from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from .models import Model
from .preprocessors import Preprocessor
from .output import Output
import numpy as np
import numpy.typing as npt

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

    def process_image(self, img: npt.NDArray[np.ubyte]) -> Output:
        """
        Perform inference on the given image

        Args:
            img (npt.NDArray[np.ubyte]): The image to perform inference on

        Returns:
            Output: The output of the model
        """
        inputs: list[npt.NDArray[np.ubyte]] = self.preprocessor.preprocess(img)
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
