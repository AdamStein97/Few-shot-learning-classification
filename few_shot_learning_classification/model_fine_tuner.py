from typing import Dict

import numpy as np
import pandas as pd
import torch
from transformers import (BartForSequenceClassification, BartTokenizer,
                          Trainer, TrainingArguments)

from few_shot_learning_classification.config import Config
from few_shot_learning_classification.utils import create_text_pairs


class MNLIDataset(torch.utils.data.Dataset):
    """
    Torch dataset for fine-tuning model
    """

    def __init__(self, encodings: np.array, labels: np.array):
        self.encodings = encodings
        self.labels = [2 if label else 0 for label in labels]

    def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class ModelFineTuner:
    """
    High-level wrapper on top of huggingface trainer to convert zero-shot model
    into few-shot model
    """

    def __init__(
        self,
        model: BartForSequenceClassification,
        model_str: str = "facebook/bart-large-mnli",
    ):
        self.tokenizer = BartTokenizer.from_pretrained(model_str)
        self.model = model

    def fine_tune(
        self,
        train_df: pd.DataFrame,
        hypothesis: str,
        num_train_epochs: int = Config.FINE_TUNE_EPOCHS,
        batch_size: int = Config.FINE_TUNE_BS,
        **kwargs
    ) -> BartForSequenceClassification:
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=Config.FINE_TUNE_LR,
        )
        train_encodings = self._prepare_data(train_df, hypothesis)
        train_dataset = MNLIDataset(train_encodings, train_df["labels"].values)

        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=train_dataset,
        )

        trainer.train()

        return self.model

    def _prepare_data(self, df: pd.DataFrame, hypothesis) -> Dict[str, np.array]:
        texts = df["text"].values
        labels = df["label"].values

        text_pairs1, text_pairs2 = create_text_pairs(hypothesis, texts, labels)

        inputs = self.tokenizer(
            text=text_pairs1,
            text_pair=text_pairs2,
            truncation="only_first",
            padding=True,
            max_length=Config.MAX_LENGTH,
        )
        return inputs
