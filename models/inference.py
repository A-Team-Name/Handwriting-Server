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

        inputs:       list[npt.NDArray[np.ubyte]] = []
        indents:      list[int]                   = []
        output_preds: list[list[str]]             = []
        output_probs: list[list[float]]           = []

        if indentation:
            inputs, indents = self.preprocessor.preprocess(img, indentation)
            print(indents)
        else:
            inputs = self.preprocessor.preprocess(img, indentation)

        for i in range(len(inputs)):
            output = self.model.predict(inputs[i])
            if indentation:
                output_preds += [
                    [
                        ("    " * indents[i]) + s
                        for s in strings
                    ]
                    for strings in output.top_preds
                ]
            else:
                output_preds += output.top_preds
            output_probs += output.top_probs
            
        return Output(
            output_preds,
            output_probs
        )
