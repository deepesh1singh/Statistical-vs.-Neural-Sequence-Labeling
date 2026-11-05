import argparse
import json
import os
import sys
from collections import Counter
import numpy as np
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import DATASETS, DEFAULT_DATA_DIR, DEFAULT_OUT, evaluate_tags, load_split, main_score

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save(fig, outdir, name):
    fig.tight_layout()
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=200)
    print("saved ->", path)

def per_class_f1(task, gold, pred):
    if task == "pos":
        tp, fp, fn = Counter(), Counter(), Counter()
        for g, p in zip(gold, pred):
            for a, b in zip(g, p):
                if a == b:
                    tp[a] += 1
                else:
                    fn[a] += 1
                    fp[b] += 1
        labels = set(tp) | set(fp) | set(fn)
        return {t: 2 * tp[t] / max(2 * tp[t] + fp[t] + fn[t], 1) for t in labels}
    from seqeval.metrics import classification_report

    rep = classification_report(gold, pred, output_dict=True)
    return {k: v["f1-score"] for k, v in rep.items() if k not in ("micro avg", "macro avg", "weighted avg")}

def confusion(gold, pred, labels):
    idx = {t: i for i, t in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)))
    for g, p in zip(gold, pred):
        for a, b in zip(g, p):
            M[idx[a], idx[b]] += 1
    return M

def plot_main(systems, scores, task, outdir, dataset):
    names = [n for n, _ in systems]
    vals = [100 * scores[n] for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(names, vals, color=COLORS[:len(names)])
    ax.set_ylim(max(0, min(vals) - 5), min(100.8, max(vals) + 2))
    ax.set_ylabel("Test accuracy (%)" if task == "pos" else "Test entity F1 (%)")
    ax.set_title("%s: Task 1 vs Task 2 vs Task 3 (test)" % dataset.upper())
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, "%.2f" % v, ha="center", va="bottom")
    ax.tick_params(axis="x", labelsize=9)
    save(fig, outdir, "01_test_comparison.png")

def plot_per_class(systems, gold, task, outdir, dataset):
    scores = {n: per_class_f1(task, gold, p) for n, p in systems}
    labels = sorted(set().union(*[set(s) for s in scores.values()]))
    x = np.arange(len(labels))
    w = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels) + 2), 4.8))
    for k, (n, _) in enumerate(systems):
        ax.bar(x - 0.4 + w / 2 + k * w, [100 * scores[n].get(l, 0.0) for l in labels], w, label=n, color=COLORS[k])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if task == "pos" else 0, ha="right" if task == "pos" else "center")
    ax.set_ylim(0, 105)
    ax.set_ylabel("F1 (%)")
    ax.set_title("%s: per-%s F1 on test" % (dataset.upper(), "tag" if task == "pos" else "entity type"))
    ax.legend(fontsize=8, loc="lower left")
    save(fig, outdir, "02_per_class_f1.png")

def plot_training(r2, r3, outdir, dataset):
    runs = [(n, r) for n, r in (("Task 2 BiLSTM-CRF", r2), ("Task 3 BiLSTM-softmax", r3)) if r and r.get("history")]
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, (n, r) in enumerate(runs):
        h = r["history"]
        ep = [e["epoch"] for e in h]
        dev = [100 * e["dev_score"] for e in h]
        axes[0].plot(ep, [e["train_loss"] for e in h], label=n, color=COLORS[k + 1])
        axes[1].plot(ep, dev, label=n, color=COLORS[k + 1])
        b = int(np.argmax(dev))
        axes[1].scatter([ep[b]], [dev[b]], marker="*", s=140, color=COLORS[k + 1], zorder=3)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss (per sentence)")
    axes[0].set_title("Training loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("dev score (%)")
    axes[1].set_title("Dev score (star = best epoch)")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend()
    save(fig, outdir, "03_training_curves.png")

def plot_alpha(r3, outdir):
    if not r3 or "dev_by_alpha" not in r3:
        return
    pairs = sorted((float(k), 100 * v) for k, v in r3["dev_by_alpha"].items())
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(xs, ys, marker="o", color=COLORS[2])
    best = r3.get("best_alpha")
    if best is not None:
        ax.axvline(best, ls="--", color="gray")
        ax.text(best, min(ys), " selected", rotation=90, va="bottom", color="gray")
    ax.set_xlabel("alpha (prior scaling)")
    ax.set_ylabel("dev score (%)")
    ax.set_title("Task 3: dev score vs alpha")
    ax.grid(alpha=0.3)
    save(fig, outdir, "04_task3_alpha_sweep.png")

def plot_oov(runs, outdir):
    runs = [(n, r) for n, r in runs if r and "test" in r and "token_acc_oov" in r["test"]]
    if not runs:
        return
    oov_rate = runs[0][1]["test"]["oov_rate"]
    labels = ["Known words", "OOV words\n(%.1f%% of tokens)" % (100 * oov_rate)]
    x = np.arange(2)
    w = 0.8 / len(runs)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, (n, r) in enumerate(runs):
        t = r["test"]
        vals = [100 * t["token_acc_known"], 100 * t["token_acc_oov"]]
        bars = ax.bar(x - 0.4 + w / 2 + k * w, vals, w, label=n, color=COLORS[k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, "%.1f" % v, ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("token accuracy (%)")
    ax.set_title("Task 1 HMM: known vs OOV tokens (test)")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, outdir, "05_task1_known_vs_oov.png")

def plot_confusions(systems, gold, outdir, dataset):
    for k, (n, pred) in enumerate(systems):
        labels = sorted({t for g in gold for t in g} | {t for p in pred for t in p})
        M = confusion(gold, pred, labels)
        R = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels) + 2),) * 2)
        im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if R[i, j] >= 0.05:
                    ax.text(j, i, "%d" % round(100 * R[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if R[i, j] > 0.5 else "black")
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")
        ax.set_title("%s - %s (row-normalised %%)" % (dataset.upper(), n), fontsize=10)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        save(fig, outdir, "06_confusion_%s.png" % n.lower().replace(" ", "_").replace("-", ""))
        off = [(M[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j]
        off.sort(reverse=True)
        print("  top confusions (%s): %s" % (n, ", ".join("%s->%s x%d" % (a, b, c) for c, a, b in off[:6])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--show", action="store_true", help="also open the figures in windows")
    args = ap.parse_args()
    task = DATASETS[args.dataset]["task"]
    d = os.path.join(args.out_dir, args.dataset)
    outdir = os.path.join(d, "plots")
    os.makedirs(outdir, exist_ok=True)

    gold = [t for _, t in load_split(args.data_dir, args.dataset, "test")]
    systems = []
    for label, fn in (("Task 1 HMM", "task1_test_pred.json"),
                      ("Task 1b HMM+lowercase", "task1_lowerfb_test_pred.json"),
                      ("Task 2 BiLSTM-CRF", "task2_test_pred.json"),
                      ("Task 3 Neural-HMM", "task3_test_pred.json")):
        p = load_json(os.path.join(d, fn))
        if p is None:
            print("skip %s: %s not found" % (label, fn))
        elif len(p) != len(gold):
            print("skip %s: %s has %d sentences, test split has %d (rerun it without --smoke)" % (
                label, fn, len(p), len(gold)))
        else:
            systems.append((label, p))
    if not systems:
        sys.exit("no valid prediction files in " + d)

    scores = {n: main_score(task, evaluate_tags(task, gold, p)) for n, p in systems}
    for n, s in scores.items():
        print("%-20s %.4f" % (n, s))

    valid = {n for n, _ in systems}
    r1 = load_json(os.path.join(d, "task1_results.json")) if "Task 1 HMM" in valid else None
    r1b = load_json(os.path.join(d, "task1_lowerfb_results.json")) if "Task 1b HMM+lowercase" in valid else None
    r2 = load_json(os.path.join(d, "task2_results.json")) if "Task 2 BiLSTM-CRF" in valid else None
    r3 = load_json(os.path.join(d, "task3_results.json")) if "Task 3 Neural-HMM" in valid else None

    plot_main(systems, scores, task, outdir, args.dataset)
    plot_per_class(systems, gold, task, outdir, args.dataset)
    plot_training(r2, r3, outdir, args.dataset)
    plot_alpha(r3, outdir)
    plot_oov([("Task 1 HMM", r1), ("Task 1b HMM+lowercase", r1b)], outdir)
    plot_confusions(systems, gold, outdir, args.dataset)
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()