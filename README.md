# Project 2: Statistical vs. Neural Sequence Labeling

One code base for all three datasets. Select with `--dataset ewt | gum | conll`.

## Path settings

| What | Default (relative to this folder) | Override |
|---|---|---|
| Prepared data | `data/<ewt|gum|conll2003>/{train,dev,test}.jsonl` | `--data_dir` or env `P2_DATA_DIR` |
| GloVe 100d | `glove/glove.6B.100d.txt` | `--glove_path` or env `P2_GLOVE` |
| Results | `outputs/<dataset>/` | `--out_dir` or env `P2_OUT_DIR` |
| Raw downloads (fallback) | `data/_raw/` | `--raw_dir` (prepare_data.py) |

## Run order (replace ewt with gum or conll)

1. `pip install -r requirements.txt`
2. `python download_glove.py`
3. `python prepare_data.py --dataset ewt`
4. `python task1_hmm.py --dataset ewt`
5. `python task2_bilstm_crf.py --dataset ewt`
6. `python task3_neural_hmm.py --dataset ewt`
7. `python compare.py --dataset ewt`

Pipeline check first: add `--smoke` to steps 4 to 6 (300 train sentences, 2 epochs).

## Outputs (outputs/<dataset>/)

- `task1_results.json`, `task2_results.json`, `task3_results.json`
- `task*_test_pred.json` (predicted tag strings for the test split)
- `task2_model.pt`
- `comparison.md`, `comparison.tex`

## Design notes

- Task 1: first-order HMM with START/STOP, add-lambda (0.1) for transitions and emissions. Unknown words: P(tag | final 3/2/1 chars + capitalization) learned from words with train frequency <= `--suffix_max_freq` (default 10), converted by dividing by P(tag); longest suffix seen wins; if none seen, uniform over open-class tags (POS: ADJ, ADV, INTJ, NOUN, PROPN, VERB, X; NER: all 9). Viterbi in log-space (NumPy).
- Task 2: char BiLSTM (25 emb, 25/dir) + 100-d GloVe (fine-tuned) -> dropout 0.5 -> word BiLSTM (100/dir) -> dropout 0.5 -> linear -> CRF (own implementation). SGD momentum 0.9, lr 0.015 / (1 + 0.05 * epoch), clip 5.0, <= 50 epochs, patience 5 on dev. Singleton words replaced by UNK with p = 0.5 during training (as in the paper code). Default batch size 8 (the spec does not fix it; `--batch_size 1` is closest to the paper code).
- Task 3: same encoder, softmax head, cross-entropy, same optimiser; temperature scaling on dev; emission = log p(tag|x) - alpha * log P(tag) with the Task 1 transitions and Viterbi unchanged; alpha chosen on dev from `--alphas`. Rows reported: softmax argmax (ablation), alpha = 1, dev-selected alpha.
- Metrics: POS = per-token accuracy; NER = entity-level F1 via seqeval.

## Data source notes

- `prepare_data.py` tries HuggingFace `load_dataset` first. Newer `datasets` versions no longer run dataset scripts, so `universal_dependencies` can fail; the script then falls back to the same treebanks from the UD GitHub repos (`.conllu`). Sentence counts of GUM can differ from the spec because GitHub master is a newer UD release; use `--ud_ref <tag>` to pin a release.
- CoNLL-2003 fallback uses a public text copy; IOB1 tags are converted to IOB2.