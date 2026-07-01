"""
Random-search hyperparameter tuning WITHOUT Ray (single-process loop).

Drives the same model.py / CONFIG as training. Use this when Ray will not start
(e.g. blocked by Windows group policy on a non-admin machine). It samples random
configs, trains each for a few epochs, and keeps the best by validation loss.

Run:
    python random_search.py --input INPUT_tune.npy --target padded_sequences_tune.npy --num-samples 40 --epochs 25 --subset 8000
"""

import argparse
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import build_models, forward_pass, compute_masked_loss, CONFIG
from functions import array_to_tensor


MAX_FLATTEN_DIM = 50000   # prune architectures whose CNN output is too large


def sample_config(rng):

    """
    Draws one random hyperparameter configuration, mirroring the Ray search space.

    ----------

    Parameters:
        rng (random.Random) - seeded random number generator.

    Returns:
        (dict) - one sampled configuration.
    """

    return {
        "cnn_depth":         rng.choice([4, 5, 6, 7]),
        "cnn_base_channels": rng.choice([8, 16, 32]),
        "cnn_kernel":        rng.choice([3, 5]),
        "cnn_freq_downsample": rng.choice([4, 5, 6]),
        "cnn_dropout":       rng.uniform(0.0, 0.3),
        "cnn_activation":    rng.choice(["elu", "gelu", "relu", "leaky_relu"]),
        "enc_hidden":        rng.choice([128, 256, 384, 512]),
        "lstm_layers":       rng.choice([1, 2, 3]),
        "enc_lstm_dropout":  rng.uniform(0.0, 0.4),
        "dec_lstm_dropout":  rng.uniform(0.0, 0.4),
        "head_ratio":        rng.choice([0.25, 0.5, 1.0]),
        "lambda_reg":        rng.choice([5, 10, 20, 40]),
        "lr":                10 ** rng.uniform(-4, np.log10(3e-3)),
        "batch_size":        rng.choice([16, 32, 64]),
        "weight_decay":      10 ** rng.uniform(-6, -3),
        "grad_clip":         rng.choice([0.5, 1.0, 5.0]),
        "optimizer":         rng.choice(["adam", "adamw"]),
    }


def to_model_cfg(s):

    """
    Translates a sampled configuration into a full model CONFIG dictionary. The frequency axis is
    downsampled (stride 2) in the first 'cnn_freq_downsample' blocks and left at stride 1 afterwards,
    so a smaller value keeps more frequency resolution; the time stride is fixed at 1.

    ----------

    Parameters:
        s (dict) - one sampled configuration.

    Returns:
        cfg (dict) - a complete model CONFIG dictionary for model.build_models.
    """

    depth = s["cnn_depth"]
    base = max(1, s["cnn_base_channels"])
    channels = [min(base * (2 ** i), 512) for i in range(depth)]
    n_down = min(s["cnn_freq_downsample"], depth)
    strides = [(2, 1)] * n_down + [(1, 1)] * (depth - n_down)
    k = s["cnn_kernel"]

    cfg = dict(CONFIG)
    cfg.update({
        "cnn_channels":     channels,
        "cnn_kernel":       (k, k),
        "cnn_padding":      (k // 2, k // 2),   # symmetric: stride-1 blocks preserve frequency, like time
        "cnn_stride":       strides,
        "cnn_dropout":      s["cnn_dropout"],
        "cnn_activation":   s["cnn_activation"],
        "enc_hidden":       s["enc_hidden"],
        "lstm_layers":      s["lstm_layers"],
        "enc_lstm_dropout": s["enc_lstm_dropout"],
        "dec_lstm_dropout": s["dec_lstm_dropout"],
        "head_ratio":       s["head_ratio"],
        "lambda_reg":       s["lambda_reg"],
    })
    return cfg


def evaluate_config(s, data, epochs, device):

    """
    Builds and trains one configuration, returning its best validation loss. Invalid or oversized
    architectures are rejected with an infinite loss.

    ----------

    Parameters:
        s (dict) - one sampled configuration.
        data (tuple) - (X_train, y_train, X_val, y_val, input_hw).
        epochs (int) - number of epochs to train.
        device (torch.device) - device to train on.

    Returns:
        best (float) - best validation loss over the epochs (inf if the config is rejected).
    """

    X_tr, y_tr, X_val, y_val, input_hw = data
    cfg = to_model_cfg(s)

    try:
        cnn, enc, dec = build_models(input_hw, cfg, device)
    except Exception:
        return float("inf")
    if enc.input_size > MAX_FLATTEN_DIM:
        return float("inf")

    params = list(cnn.parameters()) + list(enc.parameters()) + list(dec.parameters())
    Opt = torch.optim.AdamW if s["optimizer"] == "adamw" else torch.optim.Adam
    optimizer = Opt(params, lr=s["lr"], weight_decay=s["weight_decay"])

    bs = s["batch_size"]
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs, shuffle=False)

    best = float("inf")
    n_train_batches = len(train_loader)
    for epoch in range(epochs):
        cnn.train(); enc.train(); dec.train()
        train_tot = 0.0
        for b, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            out = forward_pass(xb, yb, cnn, enc, dec)
            loss, _ = compute_masked_loss(out, yb, cfg)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, s["grad_clip"]); optimizer.step()
            train_tot += loss.item()
            print(f"    epoch {epoch + 1}/{epochs}  batch {b + 1}/{n_train_batches}"
                  f"  train {train_tot / (b + 1):.4f}", end="\r")

        cnn.eval(); enc.eval(); dec.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = forward_pass(xb, yb, cnn, enc, dec)
                l, _ = compute_masked_loss(out, yb, cfg); tot += l.item(); n += 1
        v = tot / max(n, 1)
        best = min(best, v)
        print(f"    epoch {epoch + 1}/{epochs}  train {train_tot / n_train_batches:.4f}"
              f"  val {v:.4f}            ")
    return best


def load_data(input_path, target_path, subset, seed, cfg):

    """
    Loads the tuning dataset, scales the regression targets, and splits it into train and validation.

    ----------

    Parameters:
        input_path (str) - path to the tuning spectrograms.
        target_path (str) - path to the tuning padded targets.
        subset (int) - if positive and smaller than the train portion, subsample it.
        seed (int) - random seed for the split.
        cfg (dict) - configuration dictionary; supplies the scaling constants.

    Returns:
        data (tuple) - (X_train, y_train, X_val, y_val, input_hw).
    """

    X = array_to_tensor(np.load(input_path, allow_pickle=True))
    y = np.load(target_path, allow_pickle=True).astype(np.float32)
    y[:, :, 4:6] /= cfg["time_max"]
    y[:, :, 6:8] /= cfg["freq_max"]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    if subset and subset < len(X_tr):
        X_tr, y_tr = X_tr[:subset], y_tr[:subset]

    y_tr = torch.tensor(y_tr, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    return X_tr, y_tr, X_val, y_val, (X.shape[2], X.shape[3])


def main():

    """
    Parses arguments, loads the tuning data, runs the random search, prints the best configuration in
    the model CONFIG schema, and writes all trials to a CSV file.

    ----------

    Parameters:
        None

    Returns:
        None
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="INPUT_tune.npy")
    ap.add_argument("--target", default="padded_sequences_tune.npy")
    ap.add_argument("--num-samples", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--subset", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.input, args.target, args.subset, args.seed, CONFIG)
    print(f"Tuning on {len(data[0])} train / {len(data[2])} val | device {device}")

    best_loss, best_s, results = float("inf"), None, []
    for i in range(args.num_samples):
        s = sample_config(rng)
        print(f"\n[{i + 1}/{args.num_samples}] "
              f"depth={s['cnn_depth']} base_ch={s['cnn_base_channels']} k={s['cnn_kernel']} "
              f"freq_down={s['cnn_freq_downsample']} act={s['cnn_activation']} "
              f"enc_hidden={s['enc_hidden']} layers={s['lstm_layers']} head={s['head_ratio']} "
              f"lambda={s['lambda_reg']} lr={s['lr']:.2e} bs={s['batch_size']} "
              f"wd={s['weight_decay']:.1e} clip={s['grad_clip']} opt={s['optimizer']}")
        v = evaluate_config(s, data, args.epochs, device)
        print(f"  -> val_loss {v:.4f}")
        results.append((v, s))
        if v < best_loss:
            best_loss, best_s = v, s

    print("\n" + "=" * 60)
    print(f"BEST val_loss {best_loss:.4f}")
    print("Best model CONFIG (paste into model.CONFIG):")
    print(to_model_cfg(best_s))
    print("=" * 60)

    with open("tuning_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["val_loss"] + list(best_s.keys()))
        for v, s in sorted(results, key=lambda r: r[0]):
            w.writerow([v] + list(s.values()))
    print("Saved tuning_results.csv")


if __name__ == "__main__":
    main()
