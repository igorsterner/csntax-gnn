import argparse
import ast
import json
import pickle
from collections import defaultdict

import stanza
import torch
from acs.minimal_pairs.tools import alignment, translation
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--lang1", type=str, required=True)
parser.add_argument("--lang2", type=str, required=True)
parser.add_argument("--split", type=str, default="test")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    all_data = {}

    print(f"split={args.split}, lang={args.lang1}-{args.lang2}")
    ds = load_dataset(
        "igorsterner/acs-benchmark", data_dir=f"{args.lang1}-{args.lang2}", split=args.split
    )

    all_data = {}
    for row in ds:
        id = row["id"]
        all_data[str(id) + "_o"] = {
            "text": row["observed_sentence"],
            "tokens": ast.literal_eval(row["observed_tokens"]),
            "langs": ast.literal_eval(row["observed_langs"]),
        }
        all_data[str(id) + "_m"] = {
            "text": row["manipulated_sentence"],
            "tokens": ast.literal_eval(row["manipulated_tokens"]),
            "langs": ast.literal_eval(row["manipulated_langs"]),
        }
    samples_data = [{"id": id, **all_data[id]} for id in all_data]
    print(f"Working with {len(samples_data)} samples")

    #  == TRANSLATE ==

    print("Translating")
    texts = [data["text"] for data in samples_data]

    texts_lang1 = translation.madlad_translate(
        texts, "jbochi/madlad400-3b-mt", args.lang1, None, batch_size=32
    )

    texts_lang2 = translation.madlad_translate(
        texts_lang1, "jbochi/madlad400-3b-mt", args.lang2, None, batch_size=32
    )

    for i, data in enumerate(samples_data):
        data["text_lang1"] = texts_lang1[i]
        data["text_lang2"] = texts_lang2[i]

    #  == PARSE ==

    print("Parsing")

    stanza_map = {
        "zh": "zh-hans",
    }

    nlp_lang1 = stanza.Pipeline(
        stanza_map.get(args.lang1, args.lang1),
        tokenize_no_ssplit=True,
    )

    out_docs = nlp_lang1.bulk_process(texts_lang1)

    for i, data in enumerate(samples_data):

        dependencies_lang1 = [
            [head.id - 1, deprel, dep.id - 1]
            for head, deprel, dep in out_docs[i].sentences[0].dependencies
            if head.id != 0 and dep.id != 0
        ]

        data["dependencies_lang1"] = dependencies_lang1
        data["upos_lang1"] = [word.upos for word in out_docs[i].sentences[0].words]
        data["words_lang1"] = [word.text for word in out_docs[i].sentences[0].words]

    nlp_lang2 = stanza.Pipeline(
        args.lang2,
        tokenize_no_ssplit=True,
    )

    out_docs = nlp_lang2.bulk_process(texts_lang2)

    for i, data in enumerate(samples_data):

        dependencies_lang2 = [
            [head.id - 1, deprel, dep.id - 1]
            for head, deprel, dep in out_docs[i].sentences[0].dependencies
            if head.id != 0 and dep.id != 0
        ]

        data["dependencies_lang2"] = dependencies_lang2
        data["upos_lang2"] = [word.upos for word in out_docs[i].sentences[0].words]
        data["words_lang2"] = [word.text for word in out_docs[i].sentences[0].words]

    #  == ALIGN ==

    print("Aligning")

    input_cs_lang2 = [(data["tokens"], data["words_lang2"]) for data in samples_data]
    alignments_cs_lang2 = alignment.batch_align(
        input_cs_lang2,
        batch_size=32,
    )

    for data, alignments in zip(samples_data, alignments_cs_lang2):
        lang2_langs = {i: [] for i in range(len(data["words_lang2"]))}
        for i, j in alignments:
            lang2_langs[j].append(data["langs"][i])
        data["lang2_langs"] = lang2_langs
        data["alignments_cs_lang2"] = [list(x) for x in alignments]

    input_cs_lang1 = [(data["tokens"], data["words_lang1"]) for data in samples_data]
    alignments_cs_lang1 = alignment.batch_align(input_cs_lang1, batch_size=32)

    for data, alignments in zip(samples_data, alignments_cs_lang1):
        lang1_langs = {i: [] for i in range(len(data["words_lang1"]))}
        for i, j in alignments:
            lang1_langs[j].append(data["langs"][i])
        data["lang1_langs"] = lang1_langs
        data["alignments_cs_lang1"] = [list(x) for x in alignments]

    #  == SAVE ==

    output_data = defaultdict(dict)

    for data in samples_data:
        if data["id"].endswith("b"):
            id = data["id"][:-2]
            del data["id"]
            output_data[id]["manipulated"] = data
        else:
            assert data["id"].endswith("g")
            id = data["id"][:-2]
            del data["id"]
            output_data[id]["observed"] = data

    OUT_PATH = f"data/preprocessed/{args.lang1}-{args.lang2}_{args.split}.json"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(
        OUT_PATH,
        "w",
    ) as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
