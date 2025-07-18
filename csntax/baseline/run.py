import argparse
from datasets import load_dataset
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import MLPClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from data import Collate, SentenceDataset


def set_seed(seed):
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    train_loader,
    val_loaders,
    roberta,
    classifier,
    criterion,
    optimizer,
    seed,
    num_epochs=5,
    device="cpu",
):

    all_results = {epoch: {} for epoch in range(num_epochs)}

    # training
    for epoch in range(num_epochs):
        roberta.train()
        classifier.train()
        train_preds, train_labels = [], []
        for tokens1, tokens2, labels in tqdm(train_loader):

            # == CLS sent 1s ==
            out1 = roberta(**tokens1)
            x1 = out1.last_hidden_state[:, 0, :]
            # == CLS sent 2s ==
            out2 = roberta(**tokens2)
            x2 = out2.last_hidden_state[:, 0, :]
            # == CLASSIFIER to binary ==
            binary_preds = classifier(x1, x2)

            loss = criterion(binary_preds, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(binary_preds, dim=1)
            correct_labels = torch.argmax(labels, dim=1)

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(correct_labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        print(f"Epoch {epoch+1}/{num_epochs}: train A={train_acc:.4f}")

        all_results[epoch]["train"] = {"train_acc": train_acc}

        roberta.eval()
        classifier.eval()
        for val_name, val_loader in tqdm(val_loaders.items()):
            if (epoch + 1) % 10 != 0:
                continue

            val_preds, val_labels = [], []
            with torch.no_grad():
                for tokens1, tokens2, labels in val_loader:

                    labels = labels.to(device)

                    out1 = roberta(**tokens1)
                    x1 = out1.last_hidden_state[:, 0, :]
                    out2 = roberta(**tokens2)
                    x2 = out2.last_hidden_state[:, 0, :]
                    binary_preds = classifier(x1, x2)

                    preds = torch.argmax(binary_preds, dim=1)
                    correct_labels = torch.argmax(labels, dim=1)

                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(correct_labels.cpu().numpy())

            val_acc = accuracy_score(val_labels, val_preds)
            print(f"Epoch {epoch+1}/{num_epochs}: val A={val_name}): {val_acc:.4f}")
            all_results[epoch][val_name] = {"val_acc": val_acc}

            # Undo minimal pair ordering swap for later sig tests
            val_preds = [int(pred) for pred in val_preds]
            val_preds = torch.tensor(val_preds).to(correct_labels.device)
            val_labels = torch.tensor(val_labels).to(correct_labels.device)
            val_preds = torch.where(val_labels == 1, val_preds, 1 - val_preds)
            val_preds = val_preds.tolist()
            all_results[epoch][val_name]["val_preds"] = val_preds

    return all_results


def main():
    DATA_DIR = "data/preprocessed/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    seed = args.seed

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    roberta = XLMRobertaModel.from_pretrained(args.model_name).to(device)

    training_set = load_dataset("igorsterner/acs-benchmark", data_dir="de-en", split="train")
    training_set = list(training_set)

    train_dataset = SentenceDataset(training_set, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=Collate(tokenizer, device),
    )

    val_loaders = {}
    for filename in os.listdir(DATA_DIR):
        if "train" not in filename:
            evaluation_set = load_dataset("igorsterner/acs-benchmark", data_dir=filename[:5], split="test")
            evaluation_set = list(evaluation_set)

            val_dataset = SentenceDataset(evaluation_set, tokenizer)

            val_loaders[filename.replace(".jsonl", "")] = DataLoader(
                val_dataset,
                batch_size=128,
                shuffle=False,
                collate_fn=Collate(tokenizer, device),
            )

    classifier = MLPClassifier(
        input_dim=roberta.config.hidden_size, hidden_dim=roberta.config.hidden_size
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

    results = train_model(
        train_loader,
        val_loaders,
        roberta,
        classifier,
        criterion,
        optimizer,
        seed=seed,
        num_epochs=80,
        device=device,
    )

    # maybe change the path depending on where you want the model/results to be saved

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    torch.save(
        classifier.state_dict(),
        f"{args.model_name}-ft-{seed}.pt",
    )


if __name__ == "__main__":
    main()
