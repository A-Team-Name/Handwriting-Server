from flask import Flask, request, jsonify
from PIL import Image

from inference import Inferer

from torch import cuda

print("Startup")

app = Flask(__name__)

inferer = Inferer()

@app.route("/translate", methods=["POST"])
def convert_to_text():
    file = request.files["image"]

    img = Image.open(file.stream)

    response = inferer.process_image(img)

    return jsonify({'msg': response, "size": [img.width, img.height]})

@app.route("/test", methods=["GET"])
def test_gpu():
    return jsonify({'msg': cuda.is_available()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")