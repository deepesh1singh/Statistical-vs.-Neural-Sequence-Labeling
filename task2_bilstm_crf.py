import argparse
import time
import torch
from common import (DATASETS, add_common_args, evaluate_tags, get_tagset, load_all, main_score, metric_name, out_path, save_json, set_seed)
from nn_common import (BiLSTMCRF, Vocab, add_nn_args, fit, get_device, ids_to_tags, predict_crf)

def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    add_nn_args(ap)
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

    vocab = Vocab(train, [dev, test], tags, args.glove_path, use_glove=not args.no_glove)
    tr, dv, te = vocab.encode(train), vocab.encode(dev), vocab.encode(test)
    gold_dev = [t for _, t in dev]
    gold_test = [t for _, t in test]

    model = BiLSTMCRF(vocab, len(tags), args).to(device)
    print("parameters: %d" % sum(p.numel() for p in model.parameters()))

    def dev_fn(m):
        pred = ids_to_tags(predict_crf(m, dv, device), tags)
        return main_score(task, evaluate_tags(task, gold_dev, pred))

    t0 = time.time()
    history, best_dev = fit(model, tr, vocab, args, device, dev_fn)
    train_time = time.time() - t0

    dev_pred = ids_to_tags(predict_crf(model, dv, device), tags)
    test_pred = ids_to_tags(predict_crf(model, te, device), tags)
    dev_res = evaluate_tags(task, gold_dev, dev_pred)
    test_res = evaluate_tags(task, gold_test, test_pred)
    print("BEST DEV %s = %.4f" % (metric_name(task), best_dev))
    print("TEST:", {k: round(v, 4) for k, v in test_res.items()})

    save_json({
        "dataset": args.dataset, "task": task, "args": vars(args), "train_seconds": train_time,
        "n_train": len(train), "n_dev": len(dev), "n_test": len(test),
        "epochs_run": len(history), "history": history, "dev": dev_res, "test": test_res,
    }, out_path(args.out_dir, args.dataset, "task2_results.json"))
    save_json(test_pred, out_path(args.out_dir, args.dataset, "task2_test_pred.json"))
    torch.save({"state": model.state_dict(), "tags": tags},
               out_path(args.out_dir, args.dataset, "task2_model.pt"))

if __name__ == "__main__":
    main()