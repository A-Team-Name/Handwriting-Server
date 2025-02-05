from dataclasses import dataclass

@dataclass
class Output:
    top_preds: list[list[str]]
    top_probs: list[list[float]]