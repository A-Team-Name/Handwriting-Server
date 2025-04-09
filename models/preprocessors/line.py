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
    

    def preprocess(self,
        image:       npt.NDArray[np.ubyte],
        indentation: bool,
    ) -> list[npt.NDArray[np.ubyte]]:
        """
        Line separation using naive connected component analysis on columns.

        Args:
            image (npt.NDArray[np.ubyte]): The image to preprocess.
            indentation (bool):            Whether to infer indentation

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
                    components.append((start, end+1))
                start = i
                end = i
                
        # cutout each connected component
        lines   = []
        offsets = np.zeros(len(components))
        for i in range(len(components)):
            start, end = components[i]
            line = image[start : end + 1]
            lines.append(line)
            print(line[0, [0, 1, 2, 3, 4, 5]])
            if indentation:
                offsets[i] = (np.logical_not(line)
                    .sum(axis = 0)
                    .nonzero()[0][0]
                    .item()
                )

        if not indentation:
            return lines

        # essentially indents←(+\e<0,¯2-/o[⍋o])[⍋⍋o]
        e        = image.shape[1] / 20                      # less then e difference in offsets => same indentation
        i        = np.argsort(offsets)                      # i←⍋offsets
        increase = np.insert(np.diff(offsets[i]), 0, 0) > e # increase←e<0,¯2-⌿offsets[i]
        indents  = np.cumsum(increase)[np.argsort(i)]       # indents←(+\increase)[⍋i]
        indents  = [it.item() for it in indents]            # return to normal python array

        return (lines, indents)

if __name__ == "__main__":
    doctest.testmod()
