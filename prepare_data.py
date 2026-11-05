"""Builds data/<folder>/{train,dev,test}.jsonl  ->  {"tokens": [...], "tags": [...]}  (string tags).

Order tried (--source auto):
  1. HuggingFace load_dataset (as in the Datasets document)
  2. Raw files (UD GitHub .conllu for EWT/GUM; CoNLL-2003 text files for conll)

Manual placement for the raw route (if downloads are blocked):
  data/_raw/ewt/en_ewt-ud-{train,dev,test}.conllu
  data/_raw/gum/en_gum-ud-{train,dev,test}.conllu
  data/_raw/conll/{train,valid,test}.txt     (token ... NER-tag per line, blank line between sentences)
"""
import argparse
import json
import os
import urllib.request

from common import DATASETS, DEFAULT_DATA_DIR

UD_REPO = {"ewt": ("UD_English-EWT", "en_ewt"), "gum": ("UD_English-GUM", "en_gum")}
HF_UD_CFG = {"ewt": "en_ewt", "gum": "en_gum"}
CONLL_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
CONLL_BASE = "https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/"
CONLL_FILES = {"train": "train.txt", "dev": "valid.txt", "test": "test.txt"}


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("downloading", url)
    urllib.request.urlretrieve(url, dest)


def write_jsonl(path, sents):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for toks, tags in sents:
            f.write(json.dumps({"tokens": toks, "tags": tags}, ensure_ascii=False) + "\n")


# ---------------- HuggingFace route ----------------
def from_hf(dataset):
    from datasets import load_dataset

    if dataset in HF_UD_CFG:
        ds = load_dataset("universal_dependencies", HF_UD_CFG[dataset])
        names = ds["train"].features["upos"].feature.names
        m = {"train": "train", "dev": "validation", "test": "test"}
        return {k: [(list(e["tokens"]), [names[i] for i in e["upos"]]) for e in ds[v]] for k, v in m.items()}
    last = None
    for kw in ({}, {"revision": "refs/convert/parquet"}):
        try:
            ds = load_dataset("eriktks/conll2003", **kw)
            try:
                names = ds["train"].features["ner_tags"].feature.names
            except Exception:
                names = CONLL_NAMES
            m = {"train": "train", "dev": "validation", "test": "test"}
            return {k: [(list(e["tokens"]), [names[i] for i in e["ner_tags"]]) for e in ds[v]] for k, v in m.items()}
        except Exception as ex:
            last = ex
    raise last


# ---------------- raw route ----------------
def parse_conllu(path):
    sents, toks, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if toks:
                    sents.append((toks, tags))
                    toks, tags = [], []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            if "-" in cols[0] or "." in cols[0]:
                continue
            toks.append(cols[1])
            tags.append(cols[3])
    if toks:
        sents.append((toks, tags))
    return sents


def iob1_to_iob2(tags):
    out = []
    for t in tags:
        if t.startswith("I-"):
            prev = out[-1] if out else "O"
            if prev == "O" or prev[2:] != t[2:]:
                t = "B-" + t[2:]
        out.append(t)
    return out


def parse_conll_txt(path):
    sents, toks, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if toks:
                    sents.append((toks, iob1_to_iob2(tags)))
                    toks, tags = [], []
                continue
            cols = line.split()
            toks.append(cols[0])
            tags.append(cols[-1])
    if toks:
        sents.append((toks, iob1_to_iob2(tags)))
    return sents


def from_raw(dataset, raw_dir, ref):
    out = {}
    if dataset in UD_REPO:
        repo, prefix = UD_REPO[dataset]
        for split in ("train", "dev", "test"):
            fn = "%s-ud-%s.conllu" % (prefix, split)
            local = os.path.join(raw_dir, dataset, fn)
            if not os.path.isfile(local):
                download("https://raw.githubusercontent.com/UniversalDependencies/%s/%s/%s" % (repo, ref, fn), local)
            out[split] = parse_conllu(local)
        return out
    for split, fn in CONLL_FILES.items():
        local = os.path.join(raw_dir, dataset, fn)
        if not os.path.isfile(local):
            download(CONLL_BASE + fn, local)
        out[split] = parse_conll_txt(local)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--source", default="auto", choices=["auto", "hf", "raw"])
    ap.add_argument("--ud_ref", default="master", help="UD GitHub branch/tag for the raw route")
    args = ap.parse_args()
    raw_dir = args.raw_dir or os.path.join(args.data_dir, "_raw")

    splits, used = None, None
    if args.source in ("auto", "hf"):
        try:
            splits = from_hf(args.dataset)
            used = "huggingface"
        except Exception as ex:
            print("[HF route failed] %s: %s" % (type(ex).__name__, str(ex)[:300]))
            if args.source == "hf":
                raise
    if splits is None:
        splits = from_raw(args.dataset, raw_dir, args.ud_ref)
        used = "raw"

    folder = os.path.join(args.data_dir, DATASETS[args.dataset]["folder"])
    tagset = set()
    for split, sents in splits.items():
        sents = [s for s in sents if len(s[0]) > 0]
        write_jsonl(os.path.join(folder, split + ".jsonl"), sents)
        for _, tags in sents:
            tagset.update(tags)
        print("%-5s %6d sentences  %8d tokens" % (split, len(sents), sum(len(s[0]) for s in sents)))
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump({"source": used, "tagset": sorted(tagset)}, f, indent=2)
    print("source:", used)
    print("tagset (%d): %s" % (len(tagset), sorted(tagset)))
    print("written to:", folder)


if __name__ == "__main__":
    main()
