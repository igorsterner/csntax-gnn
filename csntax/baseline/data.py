import json
import random

import torch
from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(filepath, "r") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        real_sentence = item["real_sentence"]
        manipulated_sentence = item["manipulated_sentence"]
        if random.random() < 0.5:
            return real_sentence, manipulated_sentence, [1, 0]  # good first
        else:
            return manipulated_sentence, real_sentence, [0, 1]  # bad first


class Collate:
    def __init__(self, tokenizer, device, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        sentences1, sentences2, labels = zip(*batch)

        tokens1 = self.tokenizer(
            list(sentences1),
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        tokens2 = self.tokenizer(
            list(sentences2),
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )

        return (
            tokens1.to(self.device),
            tokens2.to(self.device),
            torch.tensor(labels).to(self.device),
        )
