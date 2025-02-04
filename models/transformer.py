from .inference import Inferer

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class TransformerInferer(Inferer):
    def __init__(self, processor_name, model_name):
        print("loading Processor")
        self.processor = TrOCRProcessor.from_pretrained(processor_name)
        print("loading model")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def process_image(self, img: Image.Image) -> str:
        pixel_values = self.processor(img.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]