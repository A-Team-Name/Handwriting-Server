import os
import numpy as np

from flask import Flask, request, jsonify
from PIL import Image

from models import Inferer
from models.output import Output
from models.models import HelloWorldModel
from models.models import ShapeContextsModel, LambdaCNNChar
from models.preprocessors import LinePreprocessor, CharPreprocessor

from torch import cuda

print("Startup")

app = Flask(__name__)

inferer: Inferer = Inferer(LambdaCNNChar(), CharPreprocessor())

@app.route("/translate", methods=["POST"])
def convert_to_text():
    file = request.files["image"]

    img = np.asarray(Image.open(file).convert("L"))

    response: Output = inferer.process_image(img)
    
    response_dict = {
        "top_preds": response.top_preds,
        "top_probs": response.top_probs
    }

    return jsonify(response_dict)

@app.route("/test", methods=["GET"])
def test_gpu():
    return jsonify({'msg': cuda.is_available()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=os.getenv("HANDWRITING_PORT", 5000))
