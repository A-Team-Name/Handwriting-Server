from .model import Model
from ..output import Output

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

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

    def predict(self, img: Image.Image) -> Output:
        """
        Perform inference on the given image

        Args:
            img (Image.Image): The image to perform inference on

        Returns:
            str: The output of the model
        """
        pixel_values = self.processor(img.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]