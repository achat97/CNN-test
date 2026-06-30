## One-time data build

1. **`wav_spectogram.py`** — uncomment bottom lines, set your `.wav` name. Run.
2. **`pulses.py`** — uncomment bottom lines. Run. → makes `INPUT.npy`, `TARGET.npy`
3. **`train_model.py`** — uncomment the two `save_padded_sequences` lines, run once, re-comment. → makes `padded_sequences.npy`

## Model loop

4. **Tune:** `python tune_hparams.py --num-samples 40 --epochs 25 --subset 8000` — paste the printed best config into `CONFIG` in `model.py`.
5. **Train:** `train_model.py` — uncomment the bottom `train_model(...)` call. Run. → makes a `.pth`
6. **Evaluate:** `evaluate.py` — set `CHECKPOINT` to that `.pth`, uncomment the bottom block. Run.


## Tuning on a dedicated tuning dataset

Pass the tuning files directly via the CLI — no code change needed for the filenames:

\```
python tune_hparams.py --input INPUT_tune.npy --target padded_sequences_tune.npy --num-samples 40 --epochs 25 --subset 8000
\```

One small edit in `load_data` in `tune_hparams.py`: the current version still carves out an unused test set. Replace this:

\```python
    X_train_full, _X_test, y_train_full, _y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed)
\```

with this:

\```python
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed)
\```


