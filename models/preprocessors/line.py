from PIL import Image
from .preprocessor import Preprocessor

class LinePreprocessor(Preprocessor):
    def __init__(self):
        pass

    def preprocess(self, image: Image.Image) -> list[Image.Image]:
        return [image]