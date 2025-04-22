from ..output import Output
import numpy as np
import numpy.typing as npt

class Model:
    """
    Interface class for all models
    """
    def __init__(self):
        pass
    
    def predict(self, img: npt.NDArray[np.ubyte]) -> Output:
        """
        Perform inference on the given image

        Args:
            img (npt.NDArray[np.ubyte]): The image to perform inference on

        Returns:
            Output: The output of the model (top_preds, top_probs)
        """
        assert False, "predict method not implemented in model class"
