from .model import Model
from ..output import Output

import numpy as np
import numpy.typing as npt
import subprocess

class ShapeContextsModel(Model):
    """
    Model class for matching with shape contexts.

    Inherits:
        Model: The base class for all models
    """

    def __init__(self):
        super().__init__()

    def predict(self, img: npt.NDArray[np.ubyte]) -> Output:
        with open('img.bytes', 'w') as file:
            img.tofile(file)

        with open('size.txt', 'w') as file:
            file.write(f'{img.shape[0]} {img.shape[1]}\n')

        # completed_process = subprocess.run(
        #     'dyalogscript', 'shape.apl',
        #     capture_output = True,
        # )

        return Output([], [])

