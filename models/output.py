from dataclasses import dataclass

@dataclass
class Output:
    """
    Output dataclass for the model predictions
    """
    top_preds: list[list[str]]
    top_probs: list[list[float]]