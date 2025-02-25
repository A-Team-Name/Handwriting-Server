import os
import numpy as np

from flask import Flask, request, jsonify
from PIL import Image

from models import Inferer
from models.output import Output
from models.models import TransformerModel
from models.preprocessors import LinePreprocessor

from torch import cuda

print("Startup")

app = Flask(__name__)

inferer: Inferer = Inferer(TransformerModel("MrFitzmaurice/TrOCR-Lambda-Calculus", "MrFitzmaurice/TrOCR-Lambda-Calculus"), LinePreprocessor())

@app.route("/translate", methods=["POST"])
def convert_to_text():
    print(request.files, flush=True)
    file = request.files["image"]

    img = np.asarray(Image.open(file).convert("L"))

    Image.fromarray(img).save("uploaded_img.png")

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
