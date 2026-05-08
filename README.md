# NFPCF — Neural Fair Privacy-aware Collaborative Filtering

A PyTorch implementation of Neural Collaborative Filtering (NCF) with a fairness-aware
fine-tuning stage. The model is pre-trained on MovieLens 1M ratings and then transferred
to a career recommendation task, where gender-based bias in the user embeddings is
mitigated through linear projection and a differential-fairness penalty.

## Pipeline

| Step | Script | Purpose |
| ---- | ------ | ------- |
| 1 | `001_pre_train.py`     | Pre-train NCF on MovieLens user/item interactions. |
| 2 | `002_fine_tune.py`     | Fine-tune the pre-trained model on the career task (unfair baseline). |
| 3 | `003_fine_tune_fair.py`| De-bias user embeddings and fine-tune with a differential-fairness loss (NFCF). |
| - | `000_visualize.ipynb`  | Dataset and embedding visualizations. |
| - | `004_evaluations.ipynb`| Compare HR@K, NDCG@K, and fairness metrics across runs. |

## Modules

- `models.py` — `NCF`, `MLP`, `GMF`, `MF` model definitions.
- `data.py` — Data loading and negative-sampling utilities.
- `evaluators.py` — HR@K and NDCG@K evaluation.
- `fairness_measures.py` — Differential fairness (ε-DF) and absolute unfairness.
- `constants.py` — Job/career label mapping.

## Setup

```bash
pip install -r requirements.txt
```

Place the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) files
(`movies.dat`, `ratings.dat`, `users.dat`) in `MovieLens/`. Pre-built train/test splits
live under `train-test/`; trained checkpoints are written to `output/` and metric logs
to `results/`.

## Usage

```bash
python 001_pre_train.py        # produces output/preTrained_NCF
python 002_fine_tune.py        # produces output/unfair_NCF
python 003_fine_tune_fair.py   # produces output/NFCF_lmbda_<λ>
```

A CUDA device is used automatically when available.

## References

- He et al., *Neural Collaborative Filtering* — https://arxiv.org/abs/1708.05031
- Islam et al., *NFCF* reference implementation — https://github.com/rashid-islam/nfcf
- https://github.com/guoyang9/NCF
- https://towardsdatascience.com/neural-collaborative-filtering-96cef1009401
