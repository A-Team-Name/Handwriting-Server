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
    ) -> list[tuple[str, npt.NDArray[np.ubyte], str]]:
        """
        Line separation using naive connected component analysis on columns.

        Args:
            image (npt.NDArray[np.ubyte]): The image to preprocess.
            indentation (bool):            Whether to infer indentation

        Returns:
            list[tuple[str, npt.NDArray[np.ubyte], str]]: The preprocessed images and their separating strings (newlines and indentation)

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
            if indentation:
                offsets[i] = (np.logical_not(line)
                    .sum(axis = 0)
                    .nonzero()[0][0]
                    .item()
                )

        # technically not necessary to early return as
        # offsets is initialised to all zeros so all the
        # indentation stuff will be a no-op, but better
        # safe than sorry
        if not indentation:
            return [
                ('', line, '\n')
                for line in lines
            ]

        # essentially indents←(+\e<0,¯2-/o[⍋o])[⍋⍋o]
        e        = image.shape[1] / 20                          # less then e difference in offsets => same indentation
        i        = np.argsort(offsets)                          # i is the indices to sort offsets, so offsets[i] is sorted
        increase = np.insert(np.diff(offsets[i]) > e, 0, False) # increase[j] is (offsets[i][j] - offsets[i][j - 1] > e) (and increase[0] is 0)
        indents  = np.cumsum(increase)[np.argsort(i)]           # cumsum(increase) tells us which level of indentation a line is in, then [argsort(i)] undoes the sort
        indents  = [it.item() for it in indents]                # return to normal python array

        return [
            ('\t' * indent, line, '\n')
            for (line, indent) in zip(lines, indents)
        ]

        '''
        indentation inference example:
        say

            offsets = [0, 5, 10, 6, 11, 1]:

        which you might see from writing that looks like

            foo
                 bar
                      baz
                  goo
                       hoo
             moo

        and say e is 2. then i is the indices which sorts offsets, so

            i = [0, 5, 1, 3, 2, 4]

        note then that

            offsets[i] = [0, 1, 5, 6, 10, 11]

        we can see that the offsets for each indentation level are grouped together.
        if we look at the pairwise differences from one to the next

            diff(offsets[i]) = [1, 4, 1, 4, 1]

        theres a big jump when we move from one indentation level to the next, which
        we can detect by comparing against e

            diff(offsets[i]) > e = [False, True, False, True, False]

        the least indented line is not be further indented than others, so we just
        stick a false on the beginning

            increases = insert(diff(offsets[i]) > e, 0, False)
                      = [False, False, True, False, True, False]

        each line is then indented by the number of jumps before it

            cumsum(increases) = [0, 0, 1, 1, 2, 2]

        indeed, there are two lines not indented, two lines indented by one, and two
        indented by 2. now we just need to undo the sorting to get these back in the
        right places.

            cumsum(increases)[argsort(i)] = [0, 1, 2, 1, 2, 0]

        gives us the desired indentations. if you want to know why [argsort(i)] undoes
        the sort, message asher and be ready for a lengthy explanation
        '''

if __name__ == "__main__":
    doctest.testmod()
