import json
import os
import numpy as np

from flask import Flask, request, jsonify
from PIL import Image

from models import Inferer
from models.output import Output
from models.models import ShapeContextsModel, APLCNNChar, LambdaCNNChar, PythonCNNChar, TransformerModel
from models.preprocessors import LinePreprocessor, CharPreprocessor

from torch import cuda

app = Flask(__name__)

models = {
    "shape-lambda-calculus": lambda: Inferer(ShapeContextsModel(), LinePreprocessor()),
    "shape-python3": lambda: Inferer(ShapeContextsModel(), LinePreprocessor()),
    "shape-dyalog_apl": lambda: Inferer(ShapeContextsModel(), LinePreprocessor()),
    "cnn-lambda-calculus":   lambda: Inferer(LambdaCNNChar(),      CharPreprocessor()),
    "cnn-python3":   lambda: Inferer(PythonCNNChar(),      CharPreprocessor()),
    "cnn-dyalog_apl":   lambda: Inferer(APLCNNChar(),      CharPreprocessor()),
    "trocr-lambda-calculus": lambda: Inferer(
        TransformerModel("MrFitzmaurice/TrOCR-Lambda-Calculus", "MrFitzmaurice/TrOCR-Lambda-Calculus"),
        LinePreprocessor()
    ),
    "trocr-dyalog_apl": lambda: Inferer(
        TransformerModel("MrFitzmaurice/TrOCR-APL", "MrFitzmaurice/TrOCR-APL"),
        LinePreprocessor()
    )
}

live_model_name = "cnn-lambda-calculus"
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

    response: Output = live_model.process_image(img, indentation = True)
    
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
