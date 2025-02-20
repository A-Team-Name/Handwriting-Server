import numpy as np
import numpy.typing as npt

class Preprocessor:
    """
    Interface for preprocessors.
    """
    def __init__(self):
        pass

    def preprocess(self, image: npt.NDArray[np.ubyte]) -> list[npt.NDArray[np.ubyte]]:
        """
        Preprocess an image.

        Args:
            image (npt.NDArray[np.ubyte]): The image to preprocess.

        Returns:
            list[npt.NDArray[np.ubyte]]: The preprocessed images.
        """
        pass
