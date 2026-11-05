# Project 2: Statistical vs. Neural Sequence Labeling

Part-of-speech tagging on **UD English EWT** with three systems:

| Task | System | Script |
|---|---|---|
| 1 | First-order HMM, add-lambda smoothing, suffix + capitalisation unknown-word model, log-space Viterbi | `task1_hmm.py` |
| 1b | Task 1 + lowercase fallback for unseen capitalised words (extension) | `task1_hmm.py --lower_mode` |
| 2 | Char+word BiLSTM-CRF (Lample et al., 2016, arXiv:1603.01360) | `task2_bilstm_crf.py` |
| 3 | Neural-emission HMM: calibrated BiLSTM-softmax emissions + Task 1 transitions and Viterbi | `task3_neural_hmm.py` |

The code base also supports UD English GUM (`--dataset gum`, POS) and CoNLL-2003 (`--dataset conll`, NER, entity-level F1 via seqeval). All reported results are for EWT; GUM and CoNLL-2003 are implemented but were not run.

## Results (EWT test split, 25,094 tokens)

| System | Dev acc. (%) | Test acc. (%) |
|---|---|---|
| Task 1: HMM | 90.97 | 91.32 |
| Task 1b: HMM + lowercase fallback | 91.25 | 91.75 |
| Task 2: BiLSTM-CRF | 96.07 | 96.21 |
| Task 3a: BiLSTM-softmax argmax (ablation) | 95.94 | 96.17 |
| Task 3b: Neural-emission HMM, alpha = 1 | 94.72 | 94.91 |
| Task 3c: Neural-emission HMM, alpha = 0.5 (dev-selected) | 94.92 | 95.11 |

The hybrid closes about 77% of the gap between Task 1 and Task 2. Single run, seed 42, CPU.

## Folder structure

```
project2/
  common.py              paths, data loading, metrics
  prepare_data.py        builds data/<dataset>/{train,dev,test}.jsonl
  download_glove.py      downloads glove.6B and extracts the 100d file
  task1_hmm.py           Task 1 and Task 1b
  nn_common.py           vocabulary, batching, encoder, CRF, training loop, calibration
  task2_bilstm_crf.py    Task 2
  task3_neural_hmm.py    Task 3
  compare.py             comparison table (Markdown and LaTeX)
  plot_results.py        all figures (may be named Plot.py locally)
  requirements.txt
  data/                  created by prepare_data.py
  glove/                 created by download_glove.py
  outputs/<dataset>/     results, predictions, tables, plots
```

## Path settings

| What | Default (relative to this folder) | Override |
|---|---|---|
| Prepared data | `data/<ewt, gum or conll2003>/{train,dev,test}.jsonl` | `--data_dir` or env `P2_DATA_DIR` |
| GloVe 100d | `glove/glove.6B.100d.txt` | `--glove_path` or env `P2_GLOVE` |
| Results | `outputs/<dataset>/` | `--out_dir` or env `P2_OUT_DIR` |
| Raw downloads (fallback route) | `data/_raw/` | `--raw_dir` (prepare_data.py) |

Quote paths that contain spaces, for example `--data_dir "D:\9TH SEM\NLP\data"`.

## Setup

Python 3.9 or newer (tested with Python 3.12 and PyTorch 2.6.0).

```
pip install -r requirements.txt
```

`seqeval` is only needed for the NER dataset; `datasets` is optional (see Data notes).

## Run order

```
python download_glove.py
python prepare_data.py --dataset ewt
python task1_hmm.py --dataset ewt --tune_suffix
python task1_hmm.py --dataset ewt --suffix_max_freq 10 --lower_mode auto
python task2_bilstm_crf.py --dataset ewt
python task3_neural_hmm.py --dataset ewt
python compare.py --dataset ewt
python plot_results.py --dataset ewt
```

Check the whole pipeline first on a tiny subset (300 training sentences, 2 epochs):

```
python task1_hmm.py --dataset ewt --smoke
python task2_bilstm_crf.py --dataset ewt --smoke
python task3_neural_hmm.py --dataset ewt --smoke
```

Smoke runs overwrite the files in `outputs/<dataset>/`; rerun the full commands afterwards.

If GloVe cannot be downloaded by the script, download `glove.6B.zip` manually and extract only `glove.6B.100d.txt` into `glove/`. On Windows a resumable download works well:

```
curl.exe -L -C - --retry 20 --retry-delay 5 -o glove\glove.6B.zip https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
tar -xf glove\glove.6B.zip -C glove glove.6B.100d.txt
```

## Command-line options

Common to all task scripts: `--dataset {ewt,gum,conll}`, `--data_dir`, `--out_dir`, `--seed` (default 42), `--smoke`.

| Script | Option | Default | Meaning |
|---|---|---|---|
| task1_hmm.py | `--lam` | 0.1 | add-lambda smoothing |
| task1_hmm.py | `--suffix_max_freq` | 10 | words with training frequency up to this value train the suffix model |
| task1_hmm.py | `--tune_suffix` | off | select suffix_max_freq on dev from {2, 5, 10, 20} |
| task1_hmm.py | `--lower_mode` | off | `off`, `smart`, `all` or `auto` (auto selects on dev); any value except `off` writes `task1_lowerfb_*` files |
| task2 / task3 | `--glove_path`, `--no_glove` | GloVe path / off | embedding file; `--no_glove` uses random vectors (debug only) |
| task2 / task3 | `--epochs` | 50 | maximum epochs |
| task2 / task3 | `--lr`, `--lr_decay`, `--momentum`, `--clip` | 0.015, 0.05, 0.9, 5.0 | SGD settings |
| task2 / task3 | `--patience` | 5 | early stopping on dev |
| task2 / task3 | `--batch_size` | 8 | mini-batch size (`1` is closest to the reference code) |
| task2 / task3 | `--dropout` | 0.5 | dropout on word representation and BiLSTM output |
| task2 / task3 | `--singleton_unk_prob` | 0.5 | probability of replacing a singleton word by UNK during training |
| task2 / task3 | `--device` | auto | `auto`, `cpu` or `cuda` |
| task3 only | `--alphas` | 0,0.25,0.5,0.75,1.0 | prior-scaling values tried on dev |
| plot_results.py | `--show` | off | also open the figures in windows |

## Outputs (`outputs/<dataset>/`)

| File | Content |
|---|---|
| `task1_results.json`, `task1_test_pred.json` | Task 1 metrics (dev, test, known vs unseen words) and test predictions |
| `task1_lowerfb_results.json`, `task1_lowerfb_test_pred.json` | Task 1b (lowercase fallback) |
| `task2_results.json`, `task2_test_pred.json`, `task2_model.pt` | Task 2 metrics, per-epoch history, predictions, best weights |
| `task3_results.json`, `task3_test_pred.json` | Task 3 metrics, history, calibration, alpha sweep, predictions |
| `comparison.md`, `comparison.tex` | Task 1 vs Task 2 vs Task 3 table |
| `plots/*.png` | comparison bars, per-tag F1, training curves, alpha sweep, known vs unseen words, confusion matrices |

## Method summary

**Task 1.** Bigram HMM with START and STOP states. Transitions and emissions use add-lambda (lambda = 0.1). Unseen words use P(tag | last 3/2/1 characters, capitalisation class), learned from rare training words and converted to an emission score by dividing by the tag prior; the longest suffix that was seen wins, otherwise a uniform distribution over open-class tags (ADJ, ADV, INTJ, NOUN, PROPN, VERB, X; all tags for NER). Decoding is Viterbi in log-space.

**Task 1b.** An unseen word whose lowercase form occurs in training reuses the lowercase emission before the suffix model is tried. `smart` skips mid-sentence capitalised words (likely names); `all` always applies; `auto` picks the best mode on dev.

**Task 2.** 25-d character embeddings, character BiLSTM (25 per direction), 100-d GloVe (fine-tuned), concatenation, dropout 0.5, word BiLSTM (100 per direction), dropout 0.5, linear layer, CRF (own implementation). SGD with momentum 0.9, lr 0.015 / (1 + 0.05 x epoch), gradient clipping 5.0, at most 50 epochs, early stopping on dev with patience 5.

**Task 3.** Same encoder without the CRF, softmax output trained with cross-entropy, temperature scaling fitted on dev. Emission score = log p(tag | x) - alpha x log P(tag); transitions, START/STOP scores and the Viterbi decoder are those of Task 1. alpha is selected on dev.

**Metrics.** POS: per-token accuracy. NER: entity-level F1 (seqeval). The dev split is used for early stopping and every hyperparameter choice; the test split is used for the final numbers only.

## Data notes

- `prepare_data.py` first tries `load_dataset` from Hugging Face. The `universal_dependencies` loader is not available in recent `datasets` versions (the message `[HF route failed]` is expected), so the script falls back to the official UD GitHub `.conllu` files. Multiword-token ranges and empty nodes are skipped. Sentence counts therefore differ slightly from the approximate sizes in the assignment (EWT: 12,544 / 2,001 / 2,077). Use `--ud_ref <tag>` to pin a UD release.
- For CoNLL-2003 the fallback reads a public text copy and converts IOB1 tags to IOB2.
- Neural vocabularies include dev and test words that have a GloVe vector (no labels are used).

## Runtime

On a CPU (EWT, batch size 8): Task 1 under one second; Task 2 about 2.4 hours (36 epochs); Task 3 about 1.9 hours (36 epochs). With a GPU use `--device cuda`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Missing ...jsonl` | run `prepare_data.py --dataset <name>` first |
| `GloVe file not found` | run `download_glove.py` or pass `--glove_path` |
| GloVe download times out | use the resumable `curl.exe` command above |
| pip prints dependency conflicts for streamlit or transformers | not used by this project; use a separate virtual environment to avoid them |
| `plot_results.py` prints `skip ... has N sentences` | that prediction file comes from a `--smoke` run; rerun the task without `--smoke` |
| Neural runs are too slow | use `--device cuda`, or a larger `--batch_size` (results can change slightly) |

## References

- Lample, Ballesteros, Subramanian, Kawakami, Dyer. Neural Architectures for Named Entity Recognition. NAACL 2016. arXiv:1603.01360.
- de Marneffe, Manning, Nivre, Zeman. Universal Dependencies. Computational Linguistics 47(2), 2021.
- Pennington, Socher, Manning. GloVe: Global Vectors for Word Representation. EMNLP 2014.
- Brants. TnT: A Statistical Part-of-Speech Tagger. ANLP 2000.
- Guo, Pleiss, Sun, Weinberger. On Calibration of Modern Neural Networks. ICML 2017.
