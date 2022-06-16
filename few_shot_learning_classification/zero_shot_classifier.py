import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import BartForSequenceClassification, BartTokenizer

from few_shot_learning_classification.config import Config
from few_shot_learning_classification.utils import create_text_pairs


class ZeroShotClassifier:
    def __init__(
        self,
        path_to_trained_model=None,
    ):
        if path_to_trained_model is None:
            path_to_trained_model = "facebook/bart-large-mnli"
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli")
        model = BartForSequenceClassification.from_pretrained(path_to_trained_model)
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.detach = (
            lambda x: x.cpu().detach().numpy()
            if torch.cuda.is_available()
            else x.detach().numpy()
        )

    def _predict_on_batch(
        self,
        text_batch: List[str],
        labels_batch: List[str],
        hypothesis: str,
        threshold: float = 0.99,
        max_tokenizer_length: int = Config.MAX_LENGTH,
    ) -> Tuple[List[int], List[float]]:

        text_pairs1, text_pairs2 = create_text_pairs(
            hypothesis, text_batch, labels_batch
        )

        inputs = self.tokenizer(
            text=text_pairs1,
            text_pair=text_pairs2,
            return_tensors="pt",
            truncation="only_first",
            padding=True,
            max_length=max_tokenizer_length,
        )

        inputs.to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs)[0].cpu()

        entail_contr_logits = logits[..., [0, 2]]
        probabilities = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(
            -1, keepdims=True
        )
        probabilities = self.detach(probabilities[..., 1])

        labels = probabilities > threshold
        labels = [int(label) for label in labels]

        return labels, probabilities.tolist()

    def predict(
        self,
        texts: List[str],
        labels: List[str],
        hypothesis: Optional[List[str]] = None,
        batch_size: int = Config.BART_INFERENCE_BS,
        threshold: float = 0.99,
        max_tokenizer_length: int = Config.MAX_LENGTH,
    ) -> Tuple[List[int], List[float]]:
        labels = []
        probabilities = []

        for i in tqdm(range(math.ceil(len(texts) / batch_size))):
            if i * batch_size < len(texts):
                text_batch = texts[i * batch_size: (i + 1) * batch_size]
                label_batch = labels[i * batch_size: (i + 1) * batch_size]
                batch_labels, batch_probs = self._predict_on_batch(
                    text_batch,
                    label_batch,
                    hypothesis=hypothesis,
                    threshold=threshold,
                    max_tokenizer_length=max_tokenizer_length,
                )
                labels += batch_labels
                probabilities += batch_probs
        return labels, probabilities
