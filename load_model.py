from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load the model in container
# Needed so we can have the models in the image as opposed
# to downloading them every time we start the container
TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")