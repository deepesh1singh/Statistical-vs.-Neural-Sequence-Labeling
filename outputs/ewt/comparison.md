| System | Test accuracy (%) |
|---|---|
| Task 1: HMM (count-based emissions) | 91.32 |
| Task 1b: HMM + lowercase fallback (extension) | 91.75 |
| Task 2: char+word BiLSTM-CRF | 96.21 |
| Task 3a: BiLSTM-softmax, per-token argmax (ablation) | 96.17 |
| Task 3b: neural-emission HMM (alpha=1) | 94.91 |
| Task 3c: neural-emission HMM (alpha=0.50, dev-selected) | 95.11 |
