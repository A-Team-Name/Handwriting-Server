from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class Inferer:
    def __init__(self):
        print("loading Processor")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        print("loading model")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    def process_image(self, img: Image.Image) -> str:
        pixel_values = self.processor(img.convert("RGB"), return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]