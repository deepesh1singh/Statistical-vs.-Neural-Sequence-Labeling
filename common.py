import json
import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "ewt": {"task": "pos", "folder": "ewt"},
    "gum": {"task": "pos", "folder": "gum"},
    "conll": {"task": "ner", "folder": "conll2003"},
}

DEFAULT_DATA_DIR = os.environ.get("P2_DATA_DIR", os.path.join(ROOT, "data"))
DEFAULT_GLOVE = os.environ.get("P2_GLOVE", os.path.join(ROOT, "glove", "glove.6B.100d.txt"))
DEFAULT_OUT = os.environ.get("P2_OUT_DIR", os.path.join(ROOT, "outputs"))

SMOKE_TRAIN = 300
SMOKE_EVAL = 200

def add_common_args(p):
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="tiny subset + 2 epochs, pipeline check only")

def load_split(data_dir, dataset, split, max_n=0):
    path = os.path.join(data_dir, DATASETS[dataset]["folder"], split + ".jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Missing %s\nRun: python prepare_data.py --dataset %s --data_dir %s" % (path, dataset, data_dir)
        )
    sents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if len(o["tokens"]) == 0:
                continue
            sents.append((o["tokens"], o["tags"]))
            if max_n and len(sents) >= max_n:
                break
    return sents

def load_all(args):
    n_tr = SMOKE_TRAIN if args.smoke else 0
    n_ev = SMOKE_EVAL if args.smoke else 0
    train = load_split(args.data_dir, args.dataset, "train", n_tr)
    dev = load_split(args.data_dir, args.dataset, "dev", n_ev)
    test = load_split(args.data_dir, args.dataset, "test", n_ev)
    return train, dev, test

def get_tagset(*splits):
    return sorted({t for ds in splits for _, tags in ds for t in tags})

def evaluate_tags(task, gold, pred):
    assert len(gold) == len(pred)
    if task == "pos":
        total = 0
        correct = 0
        for g, p in zip(gold, pred):
            assert len(g) == len(p)
            total += len(g)
            correct += sum(1 for a, b in zip(g, p) if a == b)
        return {"accuracy": correct / max(total, 1), "n_tokens": total}
    from seqeval.metrics import f1_score, precision_score, recall_score

    return {
        "entity_f1": float(f1_score(gold, pred)),
        "precision": float(precision_score(gold, pred)),
        "recall": float(recall_score(gold, pred)),
        "n_sentences": len(gold),
    }

def main_score(task, res):
    return res["accuracy"] if task == "pos" else res["entity_f1"]

def metric_name(task):
    return "accuracy" if task == "pos" else "entity_f1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def out_path(out_dir, dataset, name):
    d = os.path.join(out_dir, dataset)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print("saved ->", path)