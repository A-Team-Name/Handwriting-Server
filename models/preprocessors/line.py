import numpy as np
import numpy.typing as npt
if __name__ == "__main__":
    from preprocessor import Preprocessor
    import doctest
    doctest.testmod()
else:
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
        Line separation using naive connected component analysis on columns.

        Args:
            image (npt.NDArray[np.ubyte]): The image to preprocess.

        Returns:
            list[npt.NDArray[np.ubyte]]: The preprocessed images.
            
            
        >>> processor = LinePreprocessor()
        >>> test_img = np.array([[1,1,1,1,1],[1,1,1,1,1,],[1,0,0,0,1],[1,0,1,1,1],[1,1,1,1,1],[1,0,1,0,1],[1,1,1,1,1]])
        >>> processor.preprocess(test_img)
        [array([[1, 1, 1, 1, 1],
               [1, 0, 0, 0, 1],
               [1, 0, 1, 1, 1],
               [1, 1, 1, 1, 1]]), array([[1, 1, 1, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 1, 1, 1, 1]])]
        """
        # collapse each row to be min of row
        
        min_col = np.min(image, axis=1)
        
        # find connected components of value 0 and generate tuple of start and end
        # of each connected component
        
        start = 0
        end = 0
        components = []
        for i in range(len(min_col)):
            if min_col[i] == 0:
                end = i
            else:
                if start != end:
                    components.append((start+1, end))
                start = i
                end = i
                
        # add 1 pixel padding
        to_return = []
        for component in components:
            start, end = component
            to_return.append(image[start-1:end+2])
        
        return to_return

if __name__ == "__main__":
    doctest.testmod()