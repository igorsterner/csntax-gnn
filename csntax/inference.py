import json
import os

import torch
from csntax import utils 
from csntax import model as pt_model
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def test(model, test_loader, return_probs=False):
    model.eval()
    correct = 0
    total = 0
    scores = []
    if return_probs:
        probs = []

    for (graph1, graph2), labels in test_loader:
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
        labels = labels

        if return_probs:
            out_list = out.tolist()
            true_list = labels.tolist()
            for o, t in zip(out_list, true_list):
                if t[0] == 1:
                    probs.append(o)
                else:
                    o = [o[1], o[0]]
                    probs.append(o)

        pred = out.argmax(dim=1)
        true = labels.argmax(dim=1)
        correct += int((pred == true).sum())
        total += len(true)

        batch_scores = [int(s) for s in (pred == true)]
        scores.extend(batch_scores)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.1%}")

    if return_probs:
        return scores, probs
    else:
        return scores


def run_inference(model_path, test_loader, conv_type, seed, pos2idx, dep2idx):

    utils.set_seed(seed)

    model = pt_model.GraphClassifier(
        hidden_channels=10,
        node_attr_dim=1 + 3 + len(pos2idx),
        edge_attr_dim=len(dep2idx) + 1,
        conv_type=conv_type,
    )

    model.load_state_dict(torch.load(model_path))

    scores, probs = test(model, test_loader, return_probs=True)

    return scores, probs


def main():

    langs = [
        "de-en",
        "da-en",
        "es-en",
        "fr-en",
        "it-en",
        "id-en",
        "nl-en",
        "sv-en",
        "tr-en",
        "tr-de",
        "zh-en",
    ]

    for lang in langs:

        print(lang)
        for seed in [77, 123, 253]:

            with open("data/pos_tags.txt", "r") as f:
                pos_tags = f.read().splitlines()

            with open("data/relations.txt", "r") as f:
                dep_types = f.read().splitlines()

            pos2idx = {tag: i for i, tag in enumerate(pos_tags)}
            dep2idx = {dep: i for i, dep in enumerate(dep_types)}

            lang2idx = {"lang1": 0, "lang2": 1, "other": 2}

            TEST_DATA = f"data/preprocessed/{lang}_test.json"

            test_data_dict = utils.load_data(TEST_DATA, return_keys=True)
            test_data_keys, test_data = list(test_data_dict.keys()), list(
                test_data_dict.values()
            )
            test_dataset = utils.process_data(test_data, pos2idx, dep2idx, lang2idx)

            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

            model_path = f"data/models/csntax-gnn-gine-{seed}.pt"
            conv_type = "gine"

            scores, probs = run_inference(
                model_path, test_loader, conv_type, seed, pos2idx, dep2idx
            )

            os.makedirs(f"data/results/{lang}", exist_ok=True)
            with open(
                f"data/results/{lang}/csntax-gnn-{seed}_probs.json",
                "w",
            ) as f:
                json.dump(probs, f)

            print(f"({sum(scores)} out of {len(scores)} correct)")

            score_dict = {}
            for i, key in enumerate(test_data_keys):
                score_dict[key] = scores[i]

            scores_file = f"data/results/{lang}/csntax-gnn-{seed}_scores.json"
            with open(scores_file, "w") as f:
                json.dump(score_dict, f, indent=4)

if __name__ == "__main__":
    main()
