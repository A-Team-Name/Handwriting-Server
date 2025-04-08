import json
import os
import numpy as np

from flask import Flask, request, jsonify
from PIL import Image

from models import Inferer
from models.output import Output
from models.models import ShapeContextsModel, LambdaCNNChar, TransformerModel
from models.preprocessors import LinePreprocessor, CharPreprocessor

from torch import cuda

print("Startup")

app = Flask(__name__)

models = {
    "shape": lambda : Inferer(ShapeContextsModel(), LinePreprocessor()),
    "cnn": lambda : Inferer(LambdaCNNChar(), CharPreprocessor()),
    "trocr-lambda": lambda : Inferer(
        TransformerModel("MrFitzmaurice/TrOCR-Lambda-Calculus", "MrFitzmaurice/TrOCR-Lambda-Calculus"),
        CharPreprocessor()
    )
}

live_model_name = "cnn"
live_model = models[live_model_name]()

@app.route("/translate", methods=["POST"])
def convert_to_text():
    """Converts an image of handwritten text to text
    
    POST structure:
        headers: {
            "Content-Type": "multipart/form-data"
        }
        files: {
            "json": {
                "model": "model_name"
            },
            "image": image_file
        }
    

    Returns:
        _type_: _description_
    """
    if request.files.get("json") is None:
        return jsonify({"error": "No model provided"}), 400
    if request.files.get("image") is None:
        return jsonify({"error": "No image provided"}), 400
    
    model = json.loads(request.files.get("json").read().decode("utf-8"))["model"]
    file = request.files.get("image")
    
    if model in models:
        live_model = models[model]()
        live_model_name = model
    else:
        return jsonify({"error": "Model not recognized"}), 400
    img = np.asarray(Image.open(file).convert("L"))

    response: Output = live_model.process_image(img)
    
    response_dict = {
        "top_preds": response.top_preds,
        "top_probs": response.top_probs,
        "model": live_model_name,
    }

    return jsonify(response_dict)

@app.route("/test", methods=["GET"])
def test_gpu():
    return jsonify({'msg': cuda.is_available()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=os.getenv("HANDWRITING_PORT", 5000))
