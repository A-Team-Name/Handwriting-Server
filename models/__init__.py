from .inference import Inferer
from .onnx import OnnxInferer
from .transformer import TransformerInferer
from .hello_world import HelloWorldInferer

INFERER_TYPES = {
    "onnx": OnnxInferer,
    "transformer": TransformerInferer,
    "hello_world": HelloWorldInferer
}
