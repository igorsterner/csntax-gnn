# Environment

Make sure you have python installed (we used version 3.11.8) and the required dependencies.

```
conda create -n myenv python=3.11.8
conda activate myenv
pip install -r requirements.txt
```

Clone this repository and make sure the python path is set appropriately.

```
git clone https://github.com/igorsterner/csntax
cd csntax
export PYTHONPATH=$(pwd):$PYTHONPATH
```

in order to process the sentences, we use the tools from the benchmark paper. Clone that repository too

```
git clone https://github.com/igorsterner/acs
```

# CSntax-GNN

First, generate the required translations, syntax graphs and alignments for each of the languages and splits desired:

```
python csntax/data.py --lang1 de --lang2 en --split train
```

Next, train the model (by default for three different seeds):

```
python csntax/train.py
```

(add binary flags `randomize_edge_features`, `randomize_pos_tags`, `randomize_lang_features` or set `--conv_type` to `gat` for ablation)

Finally, you can evaluate on the trained language pair (de-en) and other language pairs:

```
python csntax/inference.py
```

# Baseline

Code for the finetuned XLM-R model is available in `csntax/baseline/`. Run it as follows:

```
python csntax/baseline/run.py
```

# Citation

The model is described in the following publication:

```
@inproceedings{sterner-2025-gnn,
    author = {Igor Sterner and Simone Teufel},
    title = {Code-Switching and Syntax: A Large–Scale Experiment},
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
}
```