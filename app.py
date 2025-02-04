import os
import numpy as np

from flask import Flask, request, jsonify
from PIL import Image

from models import INFERER_TYPES, Inferer

from torch import cuda

print("Startup")

app = Flask(__name__)

inferer: Inferer = INFERER_TYPES[os.getenv("INFERER_TYPE")]()

@app.route("/translate", methods=["POST"])
def convert_to_text():
    file = request.files["image"]

    img = np.asarray(Image.open(file.stream).convert("L"))

    response = inferer.process_image(img)

    return jsonify(response)

@app.route("/test", methods=["GET"])
def test_gpu():
    return jsonify({'msg': cuda.is_available()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=os.getenv("PORT", 5000))