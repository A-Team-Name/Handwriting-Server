from PIL import Image
from .preprocessor import Preprocessor

class LinePreprocessor(Preprocessor):
    """
    Preprocessor for line images.

    Inherits:
        Preprocessor: The interface for preprocessors.
    """
    def __init__(self):
        pass

    def preprocess(self, image: Image.Image) -> list[Image.Image]:
        """
        TODO: Line Separation
        Currently does nothing.

        Args:
            image (Image.Image): The image to preprocess.

        Returns:
            list[Image.Image]: The preprocessed images.
        """
        return [image]
