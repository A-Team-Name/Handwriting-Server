from .model import Model
from ..output import Output

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import numpy.typing as npt

class TransformerModel(Model):
    """
    Class to perform inference on a model with a given preprocessor

    Inherits:
        Model: The base class for performing inference
    """
    def __init__(self, processor_name: str, model_name: str):
        """
        Initialize the model with the given processor and model.
        Extra parameters are required for the processor and model names.

        Args:
            processor_name (str): Huggingface processor name
            model_name (str): Huggingface model name
        """
        self.processor = TrOCRProcessor.from_pretrained(processor_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def predict(self, img: npt.NDArray[np.ubyte]) -> Output:
        """
        Perform inference on the given image

        Args:
            img (npt.NDArray[np.ubyte]): The image to perform inference on

        Returns:
            str: The output of the model
        """
        # img is a 2d numpy array
        # Convert to RGB format
        img = np.stack((img,)*3, axis=-1)
        pixel_values = self.processor(img, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)

        text = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        chars = [[char]*3 for char in text]
        probs = [[1.0, 0.0, 0.0] for _ in text]

        return Output(chars, probs)
