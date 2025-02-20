import numpy as np
import numpy.typing as npt
from .preprocessor import Preprocessor

class LinePreprocessor(Preprocessor):
    """
    Preprocessor for line images.

    Inherits:
        Preprocessor: The interface for preprocessors.
    """
    def __init__(self):
        pass

    def preprocess(self, image: npt.NDArray[np.ubyte]) -> list[npt.NDArray[np.ubyte]]:
        """
        TODO: Line Separation
        Currently does nothing.

        Args:
            image (npt.NDArray[np.ubyte]): The image to preprocess.

        Returns:
            list[npt.NDArray[np.ubyte]]: The preprocessed images.
        """
        return [image]
