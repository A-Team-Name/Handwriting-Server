from .model import Model
from ..output import Output

import numpy as np
import numpy.typing as npt

class DummyModel(Model):
    def __init__(self):
        super().__init__()

    def predict(self, img: npt.NDArray[np.ubyte]) -> Output:
        return Output(
            [['foo', 'bar']], # predictions
            [[1.0,   0.0  ]], # probabilities
        )
