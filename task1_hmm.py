import argparse
import math
import time
from collections import Counter
import numpy as np

from common import (DATASETS, add_common_args, evaluate_tags, get_tagset, load_all, main_score, metric_name, out_path, save_json, set_seed)

OPEN_CLASS_POS = {"ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB", "X"}
LOG_EPS = math.log(1e-8)
LOWER_MODES = ("off", "smart", "all")

def cap_type(w):
    if any(c.isalpha() for c in w):
        if len(w) > 1 and w.isupper():
            return "AC"
        if w[0].isupper():
            return "IC"
        return "LC"
    return "NA"

def suffix_keys(w):
    lw = w.lower()
    cap = cap_type(w)
    return [(lw[-k:], cap) for k in (3, 2, 1) if len(lw) >= k]

def viterbi(log_start, log_trans, log_stop, log_emit):
    L, T = log_emit.shape
    if L == 0:
        return []
    delta = log_start + log_emit[0]
    back = np.zeros((L, T), dtype=np.int64)
    for i in range(1, L):
        scores = delta[:, None] + log_trans
        back[i] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + log_emit[i]
    delta = delta + log_stop
    best = int(delta.argmax())
    path = [best]
    for i in range(L - 1, 0, -1):
        best = int(back[i, best])
        path.append(best)
    path.reverse()
    return path

class HMMTagger:
    def __init__(self, tags, task, lam=0.1, suffix_max_freq=10, lower_mode="off"):
        self.tags = list(tags)
        self.T = len(self.tags)
        self.tag2i = {t: i for i, t in enumerate(self.tags)}
        self.lam = lam
        self.suffix_max_freq = suffix_max_freq
        self.lower_mode = lower_mode
        if task == "pos":
            self.open_mask = np.array([t in OPEN_CLASS_POS for t in self.tags])
        else:
            self.open_mask = np.ones(self.T, dtype=bool)
        if not self.open_mask.any():
            self.open_mask[:] = True

    def fit(self, sents):
        T, lam = self.T, self.lam
        sents = [s for s in sents if len(s[0]) > 0]
        freq = Counter(w for toks, _ in sents for w in toks)
        self.word2i = {w: i for i, w in enumerate(freq)}
        V = len(self.word2i)
        C = np.zeros((T, V))
        self.lower2i = {}
        for w in freq:
            lw = w.lower()
            if lw not in self.lower2i:
                self.lower2i[lw] = len(self.lower2i)
        CL = np.zeros((T, len(self.lower2i)))
        start = np.zeros(T)
        stop = np.zeros(T)
        trans = np.zeros((T, T))
        suf = {}
        for toks, tags in sents:
            ti = [self.tag2i[t] for t in tags]
            start[ti[0]] += 1
            stop[ti[-1]] += 1
            for a, b in zip(ti[:-1], ti[1:]):
                trans[a, b] += 1
            for w, t in zip(toks, ti):
                C[t, self.word2i[w]] += 1
                CL[t, self.lower2i[w.lower()]] += 1
                if freq[w] <= self.suffix_max_freq:
                    for key in suffix_keys(w):
                        if key not in suf:
                            suf[key] = np.zeros(T)
                        suf[key][t] += 1

        self.log_start = np.log((start + lam) / (start.sum() + lam * T))
        denom = trans.sum(axis=1) + stop + lam * (T + 1)
        self.log_trans = np.log((trans + lam) / denom[:, None])
        self.log_stop = np.log((stop + lam) / denom)

        tag_tot = C.sum(axis=1)
        self.log_emit_known = np.log((C + lam) / (tag_tot[:, None] + lam * (V + 1)))
        self.log_emit_lower = np.log((CL + lam) / (tag_tot[:, None] + lam * (V + 1)))
        self.log_prior = np.log((tag_tot + lam) / (tag_tot.sum() + lam * T))

        self.suf = {}
        for key, v in suf.items():
            self.suf[key] = np.log((v + lam) / (v.sum() + lam * T)) - self.log_prior
        u = np.full(T, LOG_EPS)
        u[self.open_mask] = -math.log(self.open_mask.sum())
        self.uniform_open = u
        self.V = V
        return self

    def unk_emission(self, w):
        for key in suffix_keys(w):
            if key in self.suf:
                return self.suf[key]
        return self.uniform_open

    def lower_index(self, w, i):
        if self.lower_mode == "off":
            return None
        k = self.lower2i.get(w.lower())
        if k is None:
            return None
        if self.lower_mode == "smart" and i > 0 and cap_type(w) == "IC":
            return None  
        return k

    def emissions(self, words):
        out = np.empty((len(words), self.T))
        for i, w in enumerate(words):
            j = self.word2i.get(w)
            if j is not None:
                out[i] = self.log_emit_known[:, j]
                continue
            k = self.lower_index(w, i)
            out[i] = self.log_emit_lower[:, k] if k is not None else self.unk_emission(w)
        return out

    def predict(self, words):
        path = viterbi(self.log_start, self.log_trans, self.log_stop, self.emissions(words))
        return [self.tags[i] for i in path]

def evaluate(model, sents, task):
    gold = [t for _, t in sents]
    pred = [model.predict(toks) for toks, _ in sents]
    res = evaluate_tags(task, gold, pred)
    kn, unk = [0, 0], [0, 0]
    low_hit = 0
    for (toks, g), p in zip(sents, pred):
        for i, (w, a, b) in enumerate(zip(toks, g, p)):
            is_known = w in model.word2i
            bucket = kn if is_known else unk
            bucket[0] += int(a == b)
            bucket[1] += 1
            if not is_known and model.lower_index(w, i) is not None:
                low_hit += 1
    res["oov_rate"] = unk[1] / max(kn[1] + unk[1], 1)
    res["token_acc_known"] = kn[0] / max(kn[1], 1)
    res["token_acc_oov"] = unk[0] / max(unk[1], 1)
    res["oov_resolved_by_lowercase"] = low_hit / max(unk[1], 1)
    return res, pred

def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--suffix_max_freq", type=int, default=10)
    ap.add_argument("--tune_suffix", action="store_true", help="pick suffix_max_freq on dev from {2,5,10,20}")
    ap.add_argument("--lower_mode", default="off", choices=["off", "smart", "all", "auto"],
                    help="lowercase fallback for unseen capitalised words; auto = pick on dev")
    args = ap.parse_args()
    set_seed(args.seed)
    task = DATASETS[args.dataset]["task"]
    train, dev, test = load_all(args)
    tags = get_tagset(train, dev, test)
    print("dataset=%s task=%s  train/dev/test = %d/%d/%d  tags=%d" % (
        args.dataset, task, len(train), len(dev), len(test), len(tags)))

    sfx = args.suffix_max_freq
    tuning = {}
    if args.tune_suffix:
        for f in (2, 5, 10, 20):
            m = HMMTagger(tags, task, args.lam, f).fit(train)
            r, _ = evaluate(m, dev, task)
            tuning[f] = main_score(task, r)
            print("  suffix_max_freq=%2d  dev %s=%.4f" % (f, metric_name(task), tuning[f]))
        sfx = max(tuning, key=tuning.get)
        print("selected suffix_max_freq =", sfx)

    mode = args.lower_mode
    lower_tuning = {}
    if mode == "auto":
        for m in LOWER_MODES:
            r, _ = evaluate(HMMTagger(tags, task, args.lam, sfx, m).fit(train), dev, task)
            lower_tuning[m] = main_score(task, r)
            print("  lower_mode=%-5s dev %s=%.4f" % (m, metric_name(task), lower_tuning[m]))
        mode = max(lower_tuning, key=lower_tuning.get)
        print("selected lower_mode =", mode)

    t0 = time.time()
    model = HMMTagger(tags, task, args.lam, sfx, mode).fit(train)
    fit_time = time.time() - t0
    dev_res, _ = evaluate(model, dev, task)
    test_res, test_pred = evaluate(model, test, task)
    print("DEV :", {k: round(v, 4) for k, v in dev_res.items()})
    print("TEST:", {k: round(v, 4) for k, v in test_res.items()})

    tag = "" if args.lower_mode == "off" else "_lowerfb"
    save_json({
        "dataset": args.dataset, "task": task, "lambda": args.lam, "suffix_max_freq": sfx,
        "suffix_tuning_dev": tuning, "lower_mode": mode, "lower_tuning_dev": lower_tuning, "fit_seconds": fit_time,
        "n_train": len(train), "n_dev": len(dev), "n_test": len(test),
        "dev": dev_res, "test": test_res,
    }, out_path(args.out_dir, args.dataset, "task1%s_results.json" % tag))
    save_json(test_pred, out_path(args.out_dir, args.dataset, "task1%s_test_pred.json" % tag))

if __name__ == "__main__":
    main()