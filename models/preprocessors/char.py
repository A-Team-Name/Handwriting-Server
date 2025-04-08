import numpy as np
import numpy.typing as npt
from .preprocessor import Preprocessor

class CharPreprocessor(Preprocessor):
    """
    Preprocessor for character images.

    Inherits:
        Preprocessor: The interface for preprocessors.
    """
    def __init__(self):
        pass

    def preprocess(self, image: npt.NDArray[np.ubyte]) -> list[npt.NDArray[np.ubyte]]:
        """
        Character separation

        Args:
            image (npt.NDArray[np.ubyte]): The line to split into characters.

        Returns:
            list[npt.NDArray[np.ubyte]]: The split images.
        """

        data = image != 255
        h, w = data.shape

        # we use a simple disjoint set data structure represented by a parent matrix
        # see https://en.wikipedia.org/wiki/Disjoint-set_data_structure
        # union just prioritises the left parent over the top one, no union by rank/size business
        # also no path splitting/merging

        # parent matrix with shape: height × width × 2 (parent row, parent column)
        p = np.zeros(
            (h, w, 2),
            dtype = np.dtype(int)
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

        min = np.zeros(len(c) - 1)                       #  leftmost pixel indices
        max = np.zeros(len(c) - 1)                       # rightmost pixel indices
        for e in range(1, len(c)):                       # fill those bad boys in
            i = np.argwhere(s == e).ravel() % w
            min[e - 1] = i.min()
            max[e - 1] = i.max()

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
            np.greater_equal.outer(min, min),
            np.logical_or(
                np.less_equal.outer(max, max),
                np.less_equal.outer(min, min + 0.5 * (max - min)),
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
        glyphs = [glyph[2] for glyph in glyphs]
        
        # pad each glyph with a border of 1 pixel
        for i in range(len(glyphs)):
            glyph = glyphs[i]
            h, w = glyph.shape
            glyph = np.pad(glyph, ((1, 1), (1, 1)), constant_values = 255)
            glyphs[i] = glyph

        return glyphs
