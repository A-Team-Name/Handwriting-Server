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

    def process_image(self,
        img:         npt.NDArray[np.ubyte],
        indentation: bool = False,
    ) -> Output:
        """
        Perform inference on the given image

        Args:
            img (npt.NDArray[np.ubyte]): The image to perform inference on
            indentation (bool):          Whether to infer indentation

        Returns:
            Output: The output of the model
        """

        output_preds: list[list[str]]             = []
        output_probs: list[list[float]]           = []

        for (prefix, image, suffix) in self.preprocessor.preprocess(img, indentation):
            for c in prefix:
                output_preds += [[c]]
                output_probs += [[1.0]]

            output = self.model.predict(image)
            output_preds += output.top_preds
            output_probs += output.top_probs

            for c in suffix:
                output_preds += [[c]]
                output_probs += [[1.0]]
            
        return Output(
            output_preds,
            output_probs
        )
