import json
import random
from collections import Counter

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from tqdm import tqdm


def load_data(path, return_keys=False):

    with open(path, "r") as file:
        data = json.load(file)

    return list(data.values()) if not return_keys else data


def build_encodings(data):

    with open("data/pos_tags.txt", "r") as f:
        pos_tags = f.read().splitlines()

    with open("data/relations.txt", "r") as f:
        dep_types = f.read().splitlines()

    print(f"POS Tags: {len(pos_tags)}")
    print(f"Dep Types: {len(dep_types)}")

    pos2idx = {tag: i for i, tag in enumerate(pos_tags)}
    dep2idx = {dep: i for i, dep in enumerate(dep_types)}
    lang2idx = {"lang1": 0, "lang2": 1, "other": 2}
    return pos2idx, dep2idx, lang2idx


def get_labels(alignment_labels, upos_labels, dependencies, data):

    alignment_labels_lang1 = alignment_labels
    labels_lang1 = ["unk"] * len(upos_labels)

    assert len(alignment_labels_lang1) == len(labels_lang1)

    for i, label_list in alignment_labels_lang1.items():

        i = int(i)

        if len(set(label_list)) == 1:
            labels_lang1[i] = label_list[0]
        elif len(label_list) > 1:
            counts = dict(Counter(label_list))
            max_count = max(counts.values())

            if list(counts.values()).count(max_count) == 1:
                labels_lang1[i] = max(counts, key=counts.get)

    num_inters = 0
    window_size = 50

    while "unk" in labels_lang1 and window_size > 0:
        for i in range(len(labels_lang1)):

            i = int(i)

            if labels_lang1[i] != "unk":
                continue

            potential_langs = []

            for head, _, rel in dependencies:
                if (
                    head == i
                    and labels_lang1[rel] != "unk"
                    and abs(head - rel) < window_size
                ):
                    potential_langs.append(labels_lang1[rel])
                elif (
                    rel == i
                    and labels_lang1[head] != "unk"
                    and abs(head - rel) < window_size
                ):
                    potential_langs.append(labels_lang1[head])

            if len(set(potential_langs)) == 1:
                labels_lang1[i] = potential_langs[0]
            elif len(potential_langs) > 1:

                counts = dict(Counter(potential_langs))
                max_count = max(counts.values())

                if list(counts.values()).count(max_count) == 1:
                    labels_lang1[i] = max(counts, key=counts.get)

        window_size -= 1

        num_inters += 1

        if num_inters > 100:
            break

    for i in range(len(labels_lang1)):
        if labels_lang1[i] not in ["lang1", "lang2"]:
            labels_lang1[i] = "other"

    return labels_lang1


def process_data(data, pos2idx, dep2idx, lang2idx):

    dataset = []

    for item in tqdm(data):

        G = nx.DiGraph()

        order = ["observed", "manipulated"]
        random.shuffle(order)

        graphs = []
        labels = []

        for acc in order:

            num_nodes_lang1 = len(item[acc]["upos_lang1"])
            num_nodes_lang2 = len(item[acc]["upos_lang2"])

            labels_lang1 = get_labels(
                item[acc]["lang1_langs"],
                item[acc]["upos_lang1"],
                item[acc]["dependencies_lang1"],
                item[acc],
            )
            labels_lang2 = get_labels(
                item[acc]["lang2_langs"],
                item[acc]["upos_lang2"],
                item[acc]["dependencies_lang2"],
                item[acc],
            )

            for i in range(num_nodes_lang1):
                node_feature = [0] * len(pos2idx)
                node_feature[pos2idx[item[acc]["upos_lang1"][i]]] = 1
                lang_feature = [0] * 3
                lang_feature[lang2idx[labels_lang1[i]]] = 1
                node_feature = lang_feature + node_feature
                node_feature.insert(0, 0)
                node_feature = torch.tensor(node_feature, dtype=torch.float32)
                G.add_node(i, x=node_feature)

            for i in range(num_nodes_lang2):
                node_feature = [0] * len(pos2idx)
                node_feature[pos2idx[item[acc]["upos_lang2"][i]]] = 1
                lang_feature = [0] * 3
                lang_feature[lang2idx[labels_lang2[i]]] = 1
                node_feature = lang_feature + node_feature
                node_feature.insert(0, 1)
                node_feature = torch.tensor(node_feature, dtype=torch.float32)
                G.add_node(i + num_nodes_lang1, x=node_feature)

            for source, relation, target in item[acc]["dependencies_lang1"]:
                relation = relation.split(":")[0]
                edge_feature = [0] * (len(dep2idx) + 1)
                edge_feature[dep2idx[relation]] = 1

                edge_feature = torch.tensor(edge_feature, dtype=torch.float32)
                G.add_edge(source, target, edge_attr=edge_feature)

            for source, relation, target in item[acc]["dependencies_lang2"]:
                relation = relation.split(":")[0]
                edge_feature = [0] * (len(dep2idx) + 1)
                edge_feature[dep2idx[relation]] = 1
                edge_feature = torch.tensor(edge_feature, dtype=torch.float32)
                G.add_edge(
                    source + num_nodes_lang1,
                    target + num_nodes_lang1,
                    edge_attr=edge_feature,
                )

            # add self connections
            for i in range(num_nodes_lang1):
                edge_feature = [0] * (len(dep2idx) + 1)
                edge_feature[-1] = 1
                edge_feature = torch.tensor(edge_feature, dtype=torch.float32)
                G.add_edge(i, i, edge_attr=edge_feature)

            for i in range(num_nodes_lang2):
                edge_feature = [0] * (len(dep2idx) + 1)
                edge_feature[-1] = 1
                edge_feature = torch.tensor(edge_feature, dtype=torch.float32)
                G.add_edge(
                    i + num_nodes_lang1, i + num_nodes_lang1, edge_attr=edge_feature
                )

            graphs.append(from_networkx(G))
            labels.append(1 if acc == "good" else 0)

        labels = torch.tensor(labels, dtype=torch.float32)
        dataset.append((tuple(graphs), labels))

    return dataset


def set_seed(seed):
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def randomize_edge_features(dataset, edge_attr_dim):
    randomized_dataset = []
    for (graph1, graph2), labels in dataset:
        for graph in [graph1, graph2]:

            random_edge_features = []
            for edge in graph.edge_attr:
                # keep self loop labels
                if edge[-1] == 1:
                    random_edge_features.append(edge)
                else:
                    rand_dep_idx = random.randint(0, edge_attr_dim - 2)
                    random_edge_feature = [0] * edge_attr_dim
                    random_edge_feature[rand_dep_idx] = 1
                    random_edge_features.append(torch.tensor(random_edge_feature))

            assert graph.edge_attr.shape == torch.stack(random_edge_features).shape

            graph.edge_attr = torch.stack(random_edge_features)

        randomized_dataset.append(((graph1, graph2), labels))
    return randomized_dataset


def randomize_pos_tags(dataset, pos2idx):
    randomized_dataset = []
    for (graph1, graph2), labels in dataset:
        for graph in [graph1, graph2]:

            random_pos_features = []
            for node in graph.x:
                lang_feature = node[:4].tolist()
                pos_feature = [0] * len(pos2idx)
                rand_pos_idx = random.randint(0, len(pos2idx) - 1)
                pos_feature[rand_pos_idx] = 1
                random_pos_features.append(torch.tensor(lang_feature + pos_feature))

            assert graph.x.shape == torch.stack(random_pos_features).shape

            graph.x = torch.stack(random_pos_features)

        randomized_dataset.append(((graph1, graph2), labels))
    return randomized_dataset


def randomize_lang_features(dataset, lang2idx):
    randomized_dataset = []
    for (graph1, graph2), labels in dataset:
        for graph in [graph1, graph2]:

            random_lang_features = []
            for node in graph.x:
                lang_feature = [0] * 3
                random_lang_idx = random.randint(0, 2)
                lang_feature[random_lang_idx] = 1

                random_node = node[:1].tolist() + lang_feature + node[4:].tolist()

                random_lang_features.append(torch.tensor(random_node))

            assert graph.x.shape == torch.stack(random_lang_features).shape

            graph.x = torch.stack(random_lang_features)

        randomized_dataset.append(((graph1, graph2), labels))
    return randomized_dataset