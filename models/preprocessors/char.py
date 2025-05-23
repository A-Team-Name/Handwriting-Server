import numpy as np
import numpy.typing as npt
from .preprocessor import Preprocessor
from .line import LinePreprocessor

class CharPreprocessor(LinePreprocessor):
    """
    Preprocessor for character images.

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
        Character separation.

        Args:
            image (npt.NDArray[np.ubyte]): The image to preprocess.
            indentation (bool):            Whether to infer indentation

        Returns:
            list[tuple[str, npt.NDArray[np.ubyte], str]]: The preprocessed images, and their separating strings
        """

        output = []

        # split by lines first, then split each by character
        for (prefix, img, suffix) in super().preprocess(image, indentation):
            data = img != 255
            h, w = data.shape

            # we use a simple disjoint set data structure represented by a parent matrix
            # see https://en.wikipedia.org/wiki/Disjoint-set_data_structure
            # union just prioritises the left parent over the top one, no union by rank/size business
            # also no path splitting/merging

            # parent matrix with shape: height × width × 2 (parent row, parent column)
            p = np.zeros(
                (h, w, 2),
                dtype = np.dtype(int),
            )
            for i in range(h):
                for j in range(w):
                    p[i, j, :] = [i, j]                                # init parent as self
                    if not data[i, j]: continue                        # if not filled here, do nothing
                    k = []                                             # potential parents
                    if i >= 1 and data[i - 1, j]: k.append((i - 1, j)) # top parent    *
                    if j >= 1 and data[i, j - 1]: k.append((i, j - 1)) # left parent  *┘
                    if len(k) == 0: continue                           # if no potential parents, leave parent as self, we are done
                    for l, (ii, jj) in enumerate(k):                   # for each potential parent indices ii jj, at index l in k    ┌──→∘
                        iii, jjj = p[ii, jj, :]                        #     get their parents                                       │ ∘─┴─∘
                        while (iii, jjj) != (ii, jj):                  #     keep stepping up parents to a root                      ×─┴─∘
                            ii, jj = iii, jjj
                            iii, jjj = p[ii, jj, :]
                        k[l] = (iii, jjj)                              #     parent of potential parent is now their root
                    p[i, j, :] = k[0]                                  # choose the root of the first parent (arbitrary) as our parent
                    if len(k) == 1: continue                           # if there was only the one potential parent, we're done
                    ii, jj = k[1]                                      # if there was a second potential parent
                    p[ii, jj, :] = k[0]                                #     merge the trees

            p = np.multiply(p, [w, 1]).sum(axis = 2).ravel() # convert to 1d parent vector
            q = p[p]                                         # find roots of each pixel
            while not np.array_equiv(p, q):
                p = q
                q = p[p]

            s = data.ravel() * (p + 1)                       # zero out non-filled nodes
            c = np.sort(np.unique(s))                        # roots
            s = np.searchsorted(c, s)                        # ids

            min_indices = np.zeros(len(c) - 1)                       #  leftmost pixel indices
            max_indices = np.zeros(len(c) - 1)                       # rightmost pixel indices
            for e in range(1, len(c)):                       # fill those bad boys in
                i = np.argwhere(s == e).ravel() % w
                min_indices[e - 1] = i.min()
                max_indices[e - 1] = i.max()

            # adjacency matrix where x → y if min(x) ≥ min(y) and (max(x) ≤ max(y) or min(x) ≤ min(y) + (max(y) - min(y))/2)
            # │# #        (  # #│       # #  )
            # │ #         (   # │        #   )
            # │#          (  #  │       #│   )
            # │      and  (     │  or    │   )
            # ││# #       ( # #││       │# # )
            # ││ #        (  # ││       │ #  )
            # ││# #       ( # #││       │# # )
            # x is either contained in the bounds of y, or, if it goes far right of y, sunk far enough into the left of y
            b = np.logical_and(
                np.greater_equal.outer(min_indices, min_indices),
                np.logical_or(
                    np.less_equal.outer(max_indices, max_indices),
                    np.less_equal.outer(min_indices, min_indices + 0.5 * (max_indices - min_indices)),
                ),
            )
            b = np.logical_or(b, b.T) # make undirected, now x ←→ y if at least one one overlaps the other
            # find transitive closure
            bb = b.dot(b)
            while not np.array_equiv(b, bb):
                b = bb
                bb = b.dot(b)

            # group together in s
            for e in range(1, len(c)):
                s[np.argwhere(s == e).ravel()] = np.nonzero(b[e - 1, :])[0][0] + 1

            # as above, find the bounds of each (group of) glyphs
            # this time put them into separate arrays
            # FIXME: it's inefficient to just do this all over, try save some results idk
            c = np.sort(np.unique(s))
            s = np.searchsorted(c, s).reshape([h, w])
            glyphs = []
            gaps = []
            for e in range(1, len(c)):
                i = np.argwhere(s == e)
                min_y, min_x = i.min(axis = 0)
                max_y, max_x = i.max(axis = 0)

                glyphs.append((
                    min_x,
                    max_x,
                    255 * np.logical_not(e == s[min_y : max_y + 1, min_x : max_x + 1]),
                ))
            glyphs.sort()
            for i in range(len(glyphs) - 1):
                gaps.append(glyphs[i + 1][0] - glyphs[i][1])
                
            max_width = max([glyph[1] - glyph[0] for glyph in glyphs])
                
            glyphs = [glyph[2] for glyph in glyphs]

            # pad each glyph with a border of 1 pixel
            for i in range(len(glyphs)):
                glyphs[i] = np.pad(glyphs[i], ((1, 1), (1, 1)), constant_values = 255)
                
            threshold = CharPreprocessor._get_space_threshold(gaps, threshold = max_width * 0.7)
            prefix_space = [' ' if gap >= threshold else '' for gap in gaps]
                
            return_glyphs = []
            for i in range(len(glyphs)):
                if i > 0:
                    return_glyphs.append((prefix_space[i-1], glyphs[i], ''))
                else:
                    return_glyphs.append(('', glyphs[i], ''))

            return_glyphs[ 0] = (prefix,        return_glyphs[ 0][1], return_glyphs[0][2])
            return_glyphs[-1] = (return_glyphs[-1][0], return_glyphs[-1][1], suffix)

            output += return_glyphs

        return output
    
    @staticmethod
    def _get_space_threshold(gaps: list[int], threshold: float) -> int:
        """
        Get the threshold for a space.

        Args:
            gaps (list[int]): The gaps between glyphs.
            threshold (float): The minimum threshold required in difference between gaps to be considered a space.

        Returns:
            int: The threshold for a space.
        """
        if len(gaps) == 0 or len(gaps) == 1:
            return 0
        
        sorted_gaps = sorted(gaps)
        differences = np.diff(sorted_gaps)
        largest_diff_index = np.argmax(differences) + 1
        gap_lower_bound = sorted_gaps[largest_diff_index] if differences[largest_diff_index - 1] > threshold else sorted_gaps[-1] + 1
        
        return gap_lower_bound
