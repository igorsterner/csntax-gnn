import argparse
import random
import os
from pathlib import Path

import torch
import wandb
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from csntax import model as pt_model
from csntax import utils

parser = argparse.ArgumentParser()
parser.add_argument("--randomize_edge_features", default=False)
parser.add_argument("--randomize_pos_tags", default=False)
parser.add_argument("--randomize_lang_features", default=False)
parser.add_argument("--conv_type", default="gine")
args = parser.parse_args()


device = "cpu"


def train_gnn(model, train_loader, optimiser, criterion):
    model.train()
    all_losses = []
    correct = 0
    total = 0
    for (
        graph1,
        graph2,
    ), labels in train_loader:
        out = model(
            graph1.x,
            graph1.edge_index,
            graph1.edge_attr,
            graph1.batch,
            graph2.x,
            graph2.edge_index,
            graph2.edge_attr,
            graph2.batch,
        )
        labels = labels.to(device)
        loss = criterion(out, labels)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        all_losses.append(loss.item())

        pred = out.argmax(dim=1)
        true = labels.argmax(dim=1)
        correct += int((pred == true).sum())
        total += len(true)

    train_acc = correct / total

    wandb.log(
        {
            "train/loss": loss.item(),
            "train/train_acc": train_acc,
        }
    )


def eval_gnn(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    all_losses = []

    for (
        graph1,
        graph2,
    ), labels in val_loader:
        out = model(
            graph1.x,
            graph1.edge_index,
            graph1.edge_attr,
            graph1.batch,
            graph2.x,
            graph2.edge_index,
            graph2.edge_attr,
            graph2.batch,
        )
        labels = labels.to(device)
        loss = criterion(out, labels)
        all_losses.append(loss.item())

        pred = out.argmax(dim=1)
        true = labels.argmax(dim=1)
        correct += int((pred == true).sum())
        total += len(true)

    val_acc = correct / total

    wandb.log(
        {
            "val/val_acc": val_acc,
        }
    )


def main():

    DATA_DIR = Path("data/preprocessed")

    TRAIN_DATA = DATA_DIR / "de-en_train.json"
    VAL_DATA = DATA_DIR / "de-en_validation.json"

    train_data = utils.load_data(TRAIN_DATA)

    pos2idx, dep2idx, lang2idx = utils.build_encodings(train_data)

    train_dataset = utils.process_data(train_data, pos2idx, dep2idx, lang2idx)

    val_data = utils.load_data(VAL_DATA)
    val_dataset = utils.process_data(val_data, pos2idx, dep2idx, lang2idx)

    random.shuffle(train_dataset)

    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")

    batch_size = 128
    learning_rate = 0.001

    if args.randomize_edge_features:
        print("Randomizing edge features")
        train_dataset = utils.randomize_edge_features(train_dataset, len(dep2idx) + 1)
        val_dataset = utils.randomize_edge_features(val_dataset, len(dep2idx) + 1)

    if args.randomize_pos_tags:
        print("Randomizing pos tags")
        train_dataset = utils.randomize_pos_tags(train_dataset, pos2idx)
        val_dataset = utils.randomize_pos_tags(val_dataset, pos2idx)

    if args.randomize_lang_features:
        print("Randomizing lang features")
        train_dataset = utils.randomize_lang_features(train_dataset, lang2idx)
        val_dataset = utils.randomize_lang_features(val_dataset, lang2idx)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    hidden_channels = 10

    for seed in [77, 123, 253]: 

        utils.set_seed(seed)

        model = pt_model.GraphClassifier(
            hidden_channels=hidden_channels,
            node_attr_dim=1 + 3 + len(pos2idx),
            edge_attr_dim=len(dep2idx) + 1,
            conv_type=args.conv_type,
        ).to(device)

        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        wandb.init(
            project="ACL2025",
            entity="igorsterner",
        )

        experiment_name = f"{args.conv_type} (seed-{seed})"

        if args.randomize_edge_features:
            experiment_name += " (random edge features)"

        if args.randomize_pos_tags:
            experiment_name += " (random pos tags)"

        if args.randomize_lang_features:
            experiment_name += " (random lang features)"

        wandb.run.name = experiment_name

        num_epochs = 100
        for epoch in tqdm(range(num_epochs)):
            train_gnn(model, train_loader, optimiser, criterion)
            eval_gnn(model, val_loader, criterion)

        MODELS_SAVE_DIR = Path("data/models/")



        if (
            args.randomize_edge_features
            or args.randomize_pos_tags
            or args.randomize_lang_features
        ):
            MODELS_SAVE_DIR = MODELS_SAVE_DIR / "ablations"

            if (
                args.randomize_edge_features
                and args.randomize_pos_tags
                and args.randomize_lang_features
            ):
                MODELS_SAVE_DIR = MODELS_SAVE_DIR / "all"
            elif args.randomize_edge_features and args.randomize_pos_tags:
                MODELS_SAVE_DIR = MODELS_SAVE_DIR / "both"
            elif args.randomize_lang_features:
                MODELS_SAVE_DIR = MODELS_SAVE_DIR / "lang"
            elif args.randomize_edge_features:
                MODELS_SAVE_DIR = MODELS_SAVE_DIR / "edge"
            elif args.randomize_pos_tags:
                MODELS_SAVE_DIR = MODELS_SAVE_DIR / "pos"
            else:
                raise Exception

        os.makedirs(MODELS_SAVE_DIR, exist_ok=True)

        torch.save(
            model.state_dict(),
            MODELS_SAVE_DIR / f"csntax-gnn-{args.conv_type}-{seed}.pt",
        )

        wandb.finish()


if __name__ == "__main__":
    main()
