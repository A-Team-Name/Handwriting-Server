from .model import Model
from ..output import Output

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import numpy.typing as npt
import torch




class TransformerModel(Model):
    """
    Class to perform inference on a model with a given preprocessor

    Inherits:
        Model: The base class for performing inference
    """
    def __init__(self, processor_name: str, model_name: str):
        """
        Initialize the model with the given processor and model.
        Extra parameters are required for the processor and model names.

        Args:
            processor_name (str): Huggingface processor name
            model_name (str): Huggingface model name
        """
        self.processor = TrOCRProcessor.from_pretrained(processor_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
    def generate_preds_and_probs(self, pixel_values):
        input_ids = self.processor.tokenizer("", return_tensors="pt").input_ids  # Start with a space token
        
        all_predictions = []
        all_probabilities = torch.zeros((1, 3), dtype=torch.float)  # Initialize all_probabilities tensor
        
        while True:
            beam_outputs = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                num_beams=3,
                num_return_sequences=3,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                top_k=3,
                max_new_tokens=1,  # Limit the length to 1 token
            )

            # select the beam with the highest score
            best_beam_index = np.argmax(beam_outputs.sequences_scores)
            # add 3 long tensor to all_predictions tensor representing the predictions of the 3 beams
            top_3_preds = torch.stack([beam_outputs.sequences[i][-1] for i in range(3)]).unsqueeze(0)
            top_3_probs = torch.stack([beam_outputs.sequences_scores[i] for i in range(3)]).unsqueeze(0)
            # softmax top_3_probs to get probabilities
            top_3_probs = torch.nn.functional.softmax(top_3_probs, dim=1)

            temp_str_predictions = []
            for i in range(top_3_preds.shape[1]):
                temp_str_predictions.append(self.processor.tokenizer.decode(top_3_preds[0][i], skip_special_tokens=True))
            temp_str_predictions = list(dict.fromkeys(temp_str_predictions))
            all_predictions.append(temp_str_predictions)  # Append the predictions to all_predictions

            all_probabilities = torch.cat([all_probabilities, top_3_probs], dim=0)  # Concatenate the probabilities
            best_beam = beam_outputs.sequences[best_beam_index]


            input_ids = torch.cat([input_ids, best_beam[-1].unsqueeze(0).unsqueeze(0)], dim=1)  # Append the new token to the input_ids

            if input_ids[0][-1] == self.processor.tokenizer.eos_token_id:
                break
            
        all_probabilities = all_probabilities[1:]
        all_probabilities = all_probabilities.squeeze(0).tolist()
            
        return all_predictions, all_probabilities

    def predict(self, img: npt.NDArray[np.ubyte]) -> Output:
        """
        Perform inference on the given image

        Args:
            img (npt.NDArray[np.ubyte]): The image to perform inference on

        Returns:
            str: The output of the model
        """
        # img is a 2d numpy array
        # Convert to RGB format
        img = img.astype(np.uint8)
        img = Image.fromarray(img).convert("RGB")
        pixel_values = self.processor(img, return_tensors="pt").pixel_values
        
        chars, probs = self.generate_preds_and_probs(pixel_values)


        return Output(chars, probs)
