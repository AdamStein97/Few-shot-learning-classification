from typing import Optional

import numpy as np
import pandas as pd

from few_shot_learning_classification.config import Config
from few_shot_learning_classification.model_fine_tuner import ModelFineTuner
from few_shot_learning_classification.zero_shot_classifier import \
    ZeroShotClassifier


class BartPredictor:
    """
    High-level class to manage inference and training of the zero-shot classifier
    """

    def __init__(self, path_to_trained_model=None, **kwargs):
        self.classifier = ZeroShotClassifier(path_to_trained_model)

    def run_inference(
        self,
        answers: np.array,
        search_term: str,
        hypothesis: Optional[str] = None,
        batch_size: int = Config.BART_INFERENCE_BS,
        threshold: float = 0.99,
        max_tokenizer_length: int = Config.MAX_LENGTH,
    ) -> pd.DataFrame:

        nodes = [search_term] * len(answers)

        hypotheses = [hypothesis] * len(answers)

        labels, probs = self.classifier.predict(
            answers,
            nodes,
            hypothesis=hypotheses,
            batch_size=batch_size,
            threshold=threshold,
            max_tokenizer_length=max_tokenizer_length,
        )
        df = pd.DataFrame(
            {"answers": answers, "pred_probability": probs, "pred_labels": labels}
        )

        return df

    def run_fine_tune(self, train_df: pd.DataFrame, **kwargs):
        fine_tuner = ModelFineTuner(self.classifier.model)
        fine_tuned_model = fine_tuner.fine_tune(train_df, **kwargs)
        self.classifier.set_model(fine_tuned_model)
