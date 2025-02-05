from PIL import Image

class Preprocessor:
    """
    Interface for preprocessors.
    """
    def __init__(self):
        pass

    def preprocess(self, image: Image.Image) -> list[Image.Image]:
        """
        Preprocess an image.

        Args:
            image (Image.Image): The image to preprocess.

        Returns:
            list[Image.Image]: The preprocessed images.
        """
        pass