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
    ) -> list[tuple[str, npt.NDArray[np.ubyte], str]]:
        """
        Preprocess an image by splitting it.

        Args:
            image (npt.NDArray[np.ubyte]): The image to preprocess.
            indentation (bool):            Whether to infer indentation

        Returns:
            list[tuple[str, npt.NDArray[np.ubyte], str]]: The preprocessed images, and their strings to be prefixed and suffixed
        """
        assert False, 'this should be overloaded'
