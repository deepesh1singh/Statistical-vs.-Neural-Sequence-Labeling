import argparse
import json
import os
from common import DATASETS, DEFAULT_OUT, metric_name

def load(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    task = DATASETS[args.dataset]["task"]
    mk = metric_name(task)
    d = os.path.join(args.out_dir, args.dataset)
    t1 = load(os.path.join(d, "task1_results.json"))
    t2 = load(os.path.join(d, "task2_results.json"))
    t3 = load(os.path.join(d, "task3_results.json"))

    rows = []
    if t1:
        rows.append(("Task 1: HMM (count-based emissions)", t1["test"]))
    t1b = load(os.path.join(d, "task1_lowerfb_results.json"))
    if t1b:
        rows.append(("Task 1b: HMM + lowercase fallback (extension)", t1b["test"]))
    if t2:
        rows.append(("Task 2: char+word BiLSTM-CRF", t2["test"]))
    if t3:
        rows.append(("Task 3a: BiLSTM-softmax, per-token argmax (ablation)", t3["test"]["argmax"]))
        rows.append(("Task 3b: neural-emission HMM (alpha=1)", t3["test"]["hmm_alpha1"]))
        rows.append(("Task 3c: neural-emission HMM (alpha=%.2f, dev-selected)" % t3["best_alpha"],
                     t3["test"]["hmm_best_alpha"]))

    if task == "pos":
        head = "| System | Test accuracy (%) |\n|---|---|\n"
        body = "".join("| %s | %.2f |\n" % (n, 100 * r[mk]) for n, r in rows)
        tex = "\\begin{tabular}{lc}\\hline System & Test acc. (\\%) \\\\\\hline\n" + "".join(
            "%s & %.2f \\\\\n" % (n, 100 * r[mk]) for n, r in rows) + "\\hline\\end{tabular}\n"
    else:
        head = "| System | Precision | Recall | Entity F1 (%) |\n|---|---|---|---|\n"
        body = "".join("| %s | %.2f | %.2f | %.2f |\n" % (n, 100 * r["precision"], 100 * r["recall"], 100 * r[mk])
                       for n, r in rows)
        tex = "\\begin{tabular}{lccc}\\hline System & P & R & F1 \\\\\\hline\n" + "".join(
            "%s & %.2f & %.2f & %.2f \\\\\n" % (n, 100 * r["precision"], 100 * r["recall"], 100 * r[mk])
            for n, r in rows) + "\\hline\\end{tabular}\n"

    table = head + body
    print(table)
    with open(os.path.join(d, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(table)
    with open(os.path.join(d, "comparison.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    print("saved ->", os.path.join(d, "comparison.md"), "and comparison.tex")

if __name__ == "__main__":
    main()