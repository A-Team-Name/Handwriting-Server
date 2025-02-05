from PIL import Image
from ..output import Output

class Model:
    def __init__(self):
        pass
    
    def predict(self, img: Image.Image) -> dict[Output, list[str]]:
        pass