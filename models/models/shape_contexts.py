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

        with open('shape-contexts.json5', 'w') as file:
            file.write( '{\n')
            file.write(f'    size:     [{img.shape[0]}, {img.shape[1]}],\n')
            file.write(f'    path:     "img.bytes",\n')
            file.write(f'    alphabet: "lc",\n')
            file.write( '}\n')

        completed_process = subprocess.run(
            ['dyalogscript', 'models/models/shape.apl'],
            capture_output = True,
            text           = True,
        )

        lines = completed_process.stdout.split('\n')[:-1]

        top_preds = list(map(list, zip(*map(list, lines))))
        #           │       │ │   │└───────┴─────────── strings to lists of chars
        #           │       │ └───┴──────────────────── transpose
        #           └───────┴────────────────────────── tuple → list

        probs = [[1.0 for _ in predset] for predset in top_preds]

        return Output(
            top_preds,
            probs,
        )

