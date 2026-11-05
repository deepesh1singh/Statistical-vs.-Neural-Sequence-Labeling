"""Shared neural code: GloVe/vocab, batching, char+word BiLSTM encoder, CRF, training loop."""
import math
import os
import random
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common import DEFAULT_GLOVE

PAD, UNK = 0, 1
DIGIT = re.compile(r"\d")


def add_nn_args(ap):
    ap.add_argument("--glove_path", default=DEFAULT_GLOVE)
    ap.add_argument("--no_glove", action="store_true", help="random word embeddings (debug only)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--lr_decay", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--singleton_unk_prob", type=float, default=0.5)
    ap.add_argument("--char_dim", type=int, default=25)
    ap.add_argument("--char_hidden", type=int, default=25)
    ap.add_argument("--word_hidden", type=int, default=100)
    ap.add_argument("--device", default="auto")


def get_device(arg):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


# ------------------------------------------------------------------ vocab / glove
def read_glove(path, needed, dim=100):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "GloVe file not found: %s\nRun: python download_glove.py   (or pass --glove_path, or --no_glove)" % path)
    vecs = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            i = line.find(" ")
            if i <= 0:
                continue
            w = line[:i]
            if w in needed:
                v = np.asarray(line[i + 1:].split(), dtype=np.float32)
                if v.shape[0] == dim:
                    vecs[w] = v
    return vecs


class Vocab:
    def __init__(self, train, others, tags, glove_path, use_glove=True, dim=100, max_chars=40):
        self.dim = dim
        self.max_chars = max_chars
        self.tags = list(tags)
        self.tag2i = {t: i for i, t in enumerate(self.tags)}
        lowers = set()
        for ds in [train] + list(others):
            for toks, _ in ds:
                for w in toks:
                    lw = w.lower()
                    lowers.add(lw)
                    lowers.add(DIGIT.sub("0", lw))
        self._glove = read_glove(glove_path, lowers, dim) if use_glove else {}

        counts = Counter(self.norm(w) for toks, _ in train for w in toks)
        self.word2i = {"<pad>": PAD, "<unk>": UNK}
        for w in counts:
            if w not in self.word2i:
                self.word2i[w] = len(self.word2i)
        for ds in others:  # dev/test words that have a pretrained vector (labels never used)
            for toks, _ in ds:
                for w in toks:
                    n = self.norm(w)
                    if n in self._glove and n not in self.word2i:
                        self.word2i[n] = len(self.word2i)
        V = len(self.word2i)
        rng = np.random.RandomState(1)
        scale = math.sqrt(3.0 / dim)
        emb = rng.uniform(-scale, scale, (V, dim)).astype(np.float32)
        emb[PAD] = 0.0
        hits = 0
        for w, i in self.word2i.items():
            if w in self._glove:
                emb[i] = self._glove[w]
                hits += 1
        self.emb = emb
        self.glove_hits = hits
        self.is_singleton = np.zeros(V, dtype=bool)
        for w, c in counts.items():
            if c == 1:
                self.is_singleton[self.word2i[w]] = True

        chars = Counter(ch for toks, _ in train for w in toks for ch in w)
        self.char2i = {"<pad>": PAD, "<unk>": UNK}
        for ch in chars:
            self.char2i[ch] = len(self.char2i)
        print("vocab: %d words (%d with GloVe vector), %d chars, %d singletons" % (
            V, hits, len(self.char2i), int(self.is_singleton.sum())))

    def norm(self, w):
        lw = w.lower()
        return lw if lw in self._glove else DIGIT.sub("0", lw)

    def encode(self, sents):
        out = []
        for toks, tags in sents:
            w = np.array([self.word2i.get(self.norm(t), UNK) for t in toks], dtype=np.int64)
            c = [[self.char2i.get(ch, UNK) for ch in t[:self.max_chars]] for t in toks]
            y = np.array([self.tag2i[t] for t in tags], dtype=np.int64)
            out.append({"w": w, "c": c, "t": y})
        return out


# ------------------------------------------------------------------ batching
def make_batches(examples, batch_size, shuffle):
    idx = list(range(len(examples)))
    if shuffle:
        random.shuffle(idx)
        span = batch_size * 50
        batches = []
        for s in range(0, len(idx), span):
            chunk = sorted(idx[s:s + span], key=lambda i: len(examples[i]["w"]))
            batches += [chunk[k:k + batch_size] for k in range(0, len(chunk), batch_size)]
        random.shuffle(batches)
        return batches
    idx.sort(key=lambda i: len(examples[i]["w"]))
    return [idx[k:k + batch_size] for k in range(0, len(idx), batch_size)]


def collate(examples, bidx, device, is_singleton=None, unk_prob=0.0):
    exs = [examples[i] for i in bidx]
    B = len(exs)
    L = max(len(e["w"]) for e in exs)
    words = np.zeros((B, L), dtype=np.int64)
    tags = np.zeros((B, L), dtype=np.int64)
    mask = np.zeros((B, L), dtype=bool)
    chars_list = []
    for b, e in enumerate(exs):
        n = len(e["w"])
        words[b, :n] = e["w"]
        tags[b, :n] = e["t"]
        mask[b, :n] = True
        chars_list.extend(e["c"])
    if unk_prob > 0 and is_singleton is not None:
        drop = is_singleton[words] & mask & (np.random.rand(B, L) < unk_prob)
        words[drop] = UNK
    C = max(len(c) for c in chars_list)
    chars = np.zeros((len(chars_list), C), dtype=np.int64)
    clen = np.zeros(len(chars_list), dtype=np.int64)
    for k, c in enumerate(chars_list):
        chars[k, :len(c)] = c
        clen[k] = len(c)
    t = lambda a: torch.from_numpy(a).to(device)
    return {"words": t(words), "tags": t(tags), "mask": t(mask), "chars": t(chars), "clen": t(clen)}


# ------------------------------------------------------------------ model
class Encoder(nn.Module):
    """char BiLSTM (25/dir) (+) GloVe(100, fine-tuned) -> dropout -> word BiLSTM (100/dir) -> dropout."""

    def __init__(self, n_chars, emb_matrix, char_dim=25, char_hidden=25, word_hidden=100, dropout=0.5):
        super().__init__()
        self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=False, padding_idx=PAD)
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=PAD)
        self.char_lstm = nn.LSTM(char_dim, char_hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.word_lstm = nn.LSTM(emb_matrix.shape[1] + 2 * char_hidden, word_hidden,
                                 batch_first=True, bidirectional=True)
        self.out_dim = 2 * word_hidden

    def forward(self, b):
        words, chars, clen, mask = b["words"], b["chars"], b["clen"], b["mask"]
        B, L = words.shape
        w = self.word_emb(words)
        packed = pack_padded_sequence(self.char_emb(chars), clen.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.char_lstm(packed)
        cvec = torch.cat([h[0], h[1]], dim=-1)
        cfull = w.new_zeros(B, L, cvec.size(-1))
        cfull[mask] = cvec
        x = self.drop(torch.cat([w, cfull], dim=-1))
        packed = pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.word_lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
        return self.drop(out)


class CRF(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.trans = nn.Parameter(torch.empty(n, n).uniform_(-0.1, 0.1))  # trans[i, j]: i -> j
        self.start = nn.Parameter(torch.empty(n).uniform_(-0.1, 0.1))
        self.end = nn.Parameter(torch.empty(n).uniform_(-0.1, 0.1))

    def gold_score(self, emis, tags, mask):
        B, L, _ = emis.shape
        score = self.start[tags[:, 0]] + emis[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for i in range(1, L):
            s = self.trans[tags[:, i - 1], tags[:, i]] + emis[:, i].gather(1, tags[:, i:i + 1]).squeeze(1)
            score = score + s * mask[:, i].float()
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return score + self.end[last]

    def log_partition(self, emis, mask):
        B, L, _ = emis.shape
        alpha = self.start.unsqueeze(0) + emis[:, 0]
        for i in range(1, L):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans.unsqueeze(0), dim=1) + emis[:, i]
            alpha = torch.where(mask[:, i].unsqueeze(1), nxt, alpha)
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def nll(self, emis, tags, mask):
        return (self.log_partition(emis, mask) - self.gold_score(emis, tags, mask)).mean()

    def decode(self, emis, mask):
        B, L, _ = emis.shape
        score = self.start.unsqueeze(0) + emis[:, 0]
        hist = []
        for i in range(1, L):
            cand = score.unsqueeze(2) + self.trans.unsqueeze(0)
            best, idx = cand.max(dim=1)
            nxt = best + emis[:, i]
            score = torch.where(mask[:, i].unsqueeze(1), nxt, score)
            hist.append(idx)
        score = score + self.end.unsqueeze(0)
        last = score.argmax(dim=1).tolist()
        lengths = mask.sum(1).tolist()
        hist = torch.stack(hist).cpu().numpy() if hist else None
        paths = []
        for b in range(B):
            tag = last[b]
            path = [tag]
            for i in range(lengths[b] - 1, 0, -1):
                tag = int(hist[i - 1][b][tag])
                path.append(tag)
            path.reverse()
            paths.append(path)
        return paths


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab, n_tags, args):
        super().__init__()
        self.enc = Encoder(len(vocab.char2i), vocab.emb, args.char_dim, args.char_hidden,
                           args.word_hidden, args.dropout)
        self.proj = nn.Linear(self.enc.out_dim, n_tags)
        self.crf = CRF(n_tags)

    def loss(self, b):
        return self.crf.nll(self.proj(self.enc(b)), b["tags"], b["mask"])

    def decode(self, b):
        return self.crf.decode(self.proj(self.enc(b)), b["mask"])


class BiLSTMSoftmax(nn.Module):
    def __init__(self, vocab, n_tags, args):
        super().__init__()
        self.enc = Encoder(len(vocab.char2i), vocab.emb, args.char_dim, args.char_hidden,
                           args.word_hidden, args.dropout)
        self.proj = nn.Linear(self.enc.out_dim, n_tags)

    def logits(self, b):
        return self.proj(self.enc(b))

    def loss(self, b):
        lg = self.logits(b)
        m = b["mask"]
        return F.cross_entropy(lg[m], b["tags"][m], reduction="sum") / lg.size(0)


# ------------------------------------------------------------------ train / predict
def fit(model, train_ex, vocab, args, device, dev_score_fn):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(args.epochs):
        lr = args.lr / (1.0 + args.lr_decay * epoch)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for bidx in make_batches(train_ex, args.batch_size, True):
            b = collate(train_ex, bidx, device, vocab.is_singleton, args.singleton_unk_prob)
            opt.zero_grad()
            loss = model.loss(b)
            loss.backward()
            nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            total += loss.item() * len(bidx)
            n += len(bidx)
        dev_score = dev_score_fn(model)
        history.append({"epoch": epoch + 1, "lr": lr, "train_loss": total / n, "dev_score": dev_score,
                        "seconds": time.time() - t0})
        print("epoch %2d  lr %.5f  loss %.4f  dev %.4f  (%.0fs)" % (
            epoch + 1, lr, total / n, dev_score, time.time() - t0), flush=True)
        if dev_score > best:
            best, bad = dev_score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping (patience %d)" % args.patience)
                break
    model.load_state_dict(best_state)
    return history, best


@torch.no_grad()
def predict_crf(model, examples, device, batch_size=64):
    model.eval()
    preds = [None] * len(examples)
    for bidx in make_batches(examples, batch_size, False):
        paths = model.decode(collate(examples, bidx, device))
        for i, p in zip(bidx, paths):
            preds[i] = p
    return preds


@torch.no_grad()
def predict_logits(model, examples, device, batch_size=64):
    model.eval()
    out = [None] * len(examples)
    for bidx in make_batches(examples, batch_size, False):
        lg = model.logits(collate(examples, bidx, device)).cpu().numpy()
        for k, i in enumerate(bidx):
            out[i] = lg[k, :len(examples[i]["w"])]
    return out


def ids_to_tags(preds, tags):
    return [[tags[i] for i in p] for p in preds]


# ------------------------------------------------------------------ calibration
def fit_temperature(logits_list, gold_list):
    z = torch.from_numpy(np.concatenate(logits_list).astype(np.float32))
    y = torch.from_numpy(np.concatenate(gold_list).astype(np.int64))
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


def np_log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def calib_stats(logits_list, gold_list, temperature, bins=15):
    z = np.concatenate(logits_list) / temperature
    y = np.concatenate(gold_list)
    lp = np_log_softmax(z)
    p = np.exp(lp)
    nll = float(-lp[np.arange(len(y)), y].mean())
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return {"nll": nll, "ece": float(ece)}
