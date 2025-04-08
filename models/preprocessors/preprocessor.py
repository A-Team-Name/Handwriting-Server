import numpy as np
import numpy.typing as npt

class Preprocessor:
    """
    Interface for preprocessors.
    """
    def __init__(self):
        pass

    def preprocess(self,
        image:       npt.NDArray[np.ubyte],
        indentation: bool,
    ) -> list[npt.NDArray[np.ubyte]]:
        """
        Preprocess an image.

        Args:
            image (npt.NDArray[np.ubyte]): The image to preprocess.
            indentation (bool):            Whether to infer indentation

        Returns:
            list[npt.NDArray[np.ubyte]]: The preprocessed images.
        """
        assert False, 'this should be overloaded'
