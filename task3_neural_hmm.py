import argparse
import time
import numpy as np
from common import (DATASETS, add_common_args, evaluate_tags, get_tagset, load_all, main_score, metric_name, out_path, save_json, set_seed)
from nn_common import (BiLSTMSoftmax, Vocab, add_nn_args, calib_stats, fit, fit_temperature, get_device, np_log_softmax, predict_logits)
from task1_hmm import HMMTagger, viterbi

def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    add_nn_args(ap)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--suffix_max_freq", type=int, default=10)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
    set_seed(args.seed)
    device = get_device(args.device)
    task = DATASETS[args.dataset]["task"]
    train, dev, test = load_all(args)
    tags = get_tagset(train, dev, test)
    print("dataset=%s task=%s device=%s  train/dev/test = %d/%d/%d  tags=%d" % (
        args.dataset, task, device, len(train), len(dev), len(test), len(tags)))

    # Task 1 HMM: transitions, start/stop and tag prior are reused unchanged
    hmm = HMMTagger(tags, task, args.lam, args.suffix_max_freq).fit(train)

    vocab = Vocab(train, [dev, test], tags, args.glove_path, use_glove=not args.no_glove)
    tr, dv, te = vocab.encode(train), vocab.encode(dev), vocab.encode(test)
    gold_dev = [t for _, t in dev]
    gold_test = [t for _, t in test]
    gold_dev_ids = [e["t"] for e in dv]
    gold_test_ids = [e["t"] for e in te]

    model = BiLSTMSoftmax(vocab, len(tags), args).to(device)
    print("parameters: %d" % sum(p.numel() for p in model.parameters()))

    def argmax_tags(logits_list):
        return [[tags[i] for i in lg.argmax(axis=1)] for lg in logits_list]

    def dev_fn(m):
        pred = argmax_tags(predict_logits(m, dv, device))
        return main_score(task, evaluate_tags(task, gold_dev, pred))

    t0 = time.time()
    history, best_dev = fit(model, tr, vocab, args, device, dev_fn)
    train_time = time.time() - t0

    lg_dev = predict_logits(model, dv, device)
    lg_test = predict_logits(model, te, device)

    temp = fit_temperature(lg_dev, gold_dev_ids)
    calib = {
        "temperature": temp,
        "dev_before": calib_stats(lg_dev, gold_dev_ids, 1.0),
        "dev_after": calib_stats(lg_dev, gold_dev_ids, temp),
        "test_before": calib_stats(lg_test, gold_test_ids, 1.0),
        "test_after": calib_stats(lg_test, gold_test_ids, temp),
    }
    print("temperature = %.3f" % temp)
    print("calibration:", calib)

    def hmm_decode(logits_list, alpha):
        out = []
        for lg in logits_list:
            emit = np_log_softmax(lg / temp) - alpha * hmm.log_prior[None, :]
            path = viterbi(hmm.log_start, hmm.log_trans, hmm.log_stop, emit)
            out.append([tags[i] for i in path])
        return out

    alphas = [float(a) for a in args.alphas.split(",")]
    dev_by_alpha = {}
    for a in alphas:
        dev_by_alpha[a] = main_score(task, evaluate_tags(task, gold_dev, hmm_decode(lg_dev, a)))
        print("  alpha=%.2f  dev %s = %.4f" % (a, metric_name(task), dev_by_alpha[a]))
    best_alpha = max(dev_by_alpha, key=dev_by_alpha.get)
    print("selected alpha (dev) =", best_alpha)

    test_argmax = evaluate_tags(task, gold_test, argmax_tags(lg_test))
    pred_a1 = hmm_decode(lg_test, 1.0)
    test_a1 = evaluate_tags(task, gold_test, pred_a1)
    pred_best = hmm_decode(lg_test, best_alpha)
    test_best = evaluate_tags(task, gold_test, pred_best)
    print("TEST softmax-argmax       :", {k: round(v, 4) for k, v in test_argmax.items()})
    print("TEST neural-HMM alpha=1   :", {k: round(v, 4) for k, v in test_a1.items()})
    print("TEST neural-HMM alpha=%.2f:" % best_alpha, {k: round(v, 4) for k, v in test_best.items()})

    save_json({
        "dataset": args.dataset, "task": task, "args": vars(args), "train_seconds": train_time,
        "epochs_run": len(history), "history": history, "calibration": calib,
        "dev_by_alpha": {str(k): v for k, v in dev_by_alpha.items()}, "best_alpha": best_alpha,
        "test": {"argmax": test_argmax, "hmm_alpha1": test_a1, "hmm_best_alpha": test_best},
    }, out_path(args.out_dir, args.dataset, "task3_results.json"))
    save_json(pred_best, out_path(args.out_dir, args.dataset, "task3_test_pred.json"))

if __name__ == "__main__":
    main()