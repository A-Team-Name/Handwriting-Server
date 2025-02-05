from PIL import Image
from ..output import Output

class Model:
    """
    Interface class for all models
    """
    def __init__(self):
        pass
    
    def predict(self, img: Image.Image) -> Output:
        """
        Perform inference on the given image

        Args:
            img (Image.Image): The image to perform inference on

        Returns:
            Output: The output of the model (top_preds, top_probs)
        """
        pass