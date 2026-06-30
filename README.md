## One-time data build

1. **`wav_spectogram.py`** — uncomment bottom lines, set your `.wav` name. Run.
2. **`pulses.py`** — uncomment bottom lines. Run. → makes `INPUT.npy`, `TARGET.npy`
3. **`train_model.py`** — uncomment the two `save_padded_sequences` lines, run once, re-comment. → makes `padded_sequences.npy`

## Model loop

4. **Tune:** `python tune_hparams.py --num-samples 40 --epochs 25 --subset 8000` — paste the printed best config into `CONFIG` in `model.py`.
5. **Train:** `train_model.py` — uncomment the bottom `train_model(...)` call. Run. → makes a `.pth`
6. **Evaluate:** `evaluate.py` — set `CHECKPOINT` to that `.pth`, uncomment the bottom block. Run.
