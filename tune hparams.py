"""
Random-search hyperparameter tuning for the CNN-LSTM encoder-decoder with Ray Tune.

This script drives the SAME model.py / CONFIG used by train_model.py and evaluate.py: it samples a
set of knobs, translates them into the model CONFIG schema, and calls model.build_models(). There is
no duplicate model definition here, so the search can never drift from the model you actually train.

The frequency axis is the one worth searching: time resolution (stride 1 on the time axis) already
gives tight time estimates, whereas downsampling frequency by a factor of two in every block is the
suspected cause of the LFM/HFM confusion and the wider frequency-regression scatter. The time stride
is therefore fixed at 1, and the number of blocks that halve the frequency axis is the tunable knob.

Run:
    python tune_hparams.py --num-samples 40 --epochs 25 --subset 8000 --gpu-per-trial 1
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from model import build_models, forward_pass, compute_masked_loss, CONFIG
from functions import array_to_tensor


# If the flattened CNN feature dimension exceeds this, the encoder LSTM becomes
# too large to be worth a trial; such configs are pruned with a large loss.
MAX_FLATTEN_DIM = 50000


def to_model_cfg(sampled):

    """
    Translates one sampled hyperparameter configuration into a full model CONFIG dictionary. The CNN
    channels are built by doubling from a base width over the chosen depth. The frequency axis is
    downsampled (stride 2) in the first 'cnn_freq_downsample' blocks and left at stride 1 in the rest,
    so a smaller value of 'cnn_freq_downsample' preserves more frequency resolution. The time stride is
    fixed at 1, since time resolution is already adequate.

    ----------

    Parameters:
        sampled (dict) - one configuration drawn from the search space.

    Returns:
        cfg (dict) - a complete model CONFIG dictionary ready for model.build_models.
    """

    depth = sampled["cnn_depth"]
    base = max(1, sampled["cnn_base_channels"])
    channels = [min(base * (2 ** i), 512) for i in range(depth)]

    # number of blocks that halve the frequency axis (clamped to the depth)
    n_down = min(sampled["cnn_freq_downsample"], depth)
    strides = [(2, 1)] * n_down + [(1, 1)] * (depth - n_down)

    k = sampled["cnn_kernel"]

    cfg = dict(CONFIG)
    cfg.update({
        "cnn_channels":     channels,
        "cnn_kernel":       (k, k),
        "cnn_padding":      (k // 2, k // 2),   # symmetric: stride-1 blocks preserve frequency, like time
        "cnn_stride":       strides,
        "cnn_dropout":      sampled["cnn_dropout"],
        "cnn_activation":   sampled["cnn_activation"],
        "enc_hidden":       sampled["enc_hidden"],
        "lstm_layers":      sampled["lstm_layers"],
        "enc_lstm_dropout": sampled["enc_lstm_dropout"],
        "dec_lstm_dropout": sampled["dec_lstm_dropout"],
        "head_ratio":       sampled["head_ratio"],
        "lambda_reg":       sampled["lambda_reg"],
    })
    return cfg


def search_space():

    """
    Defines the random-search space over the tunable hyperparameters.

    ----------

    Parameters:
        None

    Returns:
        space (dict) - mapping of hyperparameter names to Ray Tune samplers.
    """

    return {
        # CNN
        "cnn_depth":         tune.choice([4, 5, 6, 7]),
        "cnn_base_channels": tune.choice([8, 16, 32]),
        "cnn_kernel":        tune.choice([3, 5]),
        "cnn_freq_downsample": tune.choice([4, 5, 6]),  # blocks that halve frequency; lower = finer
        "cnn_dropout":       tune.uniform(0.0, 0.3),
        "cnn_activation":    tune.choice(["elu", "gelu", "relu", "leaky_relu"]),
        # LSTMs
        "enc_hidden":        tune.choice([128, 256, 384, 512]),
        "lstm_layers":       tune.choice([1, 2, 3]),
        "enc_lstm_dropout":  tune.uniform(0.0, 0.4),
        "dec_lstm_dropout":  tune.uniform(0.0, 0.4),
        "head_ratio":        tune.choice([0.25, 0.5, 1.0]),
        "lambda_reg":        tune.choice([5, 10, 20, 40]),
        # training-only knobs (not part of the model CONFIG)
        "lr":                tune.loguniform(1e-4, 3e-3),
        "batch_size":        tune.choice([16, 32, 64]),
        "weight_decay":      tune.loguniform(1e-6, 1e-3),
        "grad_clip":         tune.choice([0.5, 1.0, 5.0]),
        "optimizer":         tune.choice(["adam", "adamw"]),
    }


def train_trainable(sampled, data=None, epochs=25, seed=0):

    """
    Ray Tune trainable. Builds the model for one sampled configuration, trains it for a number of
    epochs on the training split, and reports the validation loss after every epoch so the ASHA
    scheduler can stop weak trials early. Configurations whose architecture is invalid (the frequency
    axis collapses) or too large (the flattened feature dimension exceeds MAX_FLATTEN_DIM) are pruned
    by reporting a large loss.

    ----------

    Parameters:
        sampled (dict) - one configuration drawn from the search space.
        data (tuple) - (X_train, y_train, X_val, y_val, input_hw), passed via tune.with_parameters.
        epochs (int) - maximum number of epochs per trial.
        seed (int) - random seed for reproducibility within a trial.

    Returns:
        None
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, y_tr, X_val, y_val, input_hw = data
    cfg = to_model_cfg(sampled)

    try:
        encoder_cnn, encoder_lstm, decoder = build_models(input_hw, cfg, device)
    except Exception:
        tune.report({"val_loss": 1e9})
        return

    if encoder_lstm.input_size > MAX_FLATTEN_DIM:
        tune.report({"val_loss": 1e9})
        return

    params = (list(encoder_cnn.parameters())
              + list(encoder_lstm.parameters())
              + list(decoder.parameters()))
    if sampled["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=sampled["lr"], weight_decay=sampled["weight_decay"])
    else:
        optimizer = torch.optim.Adam(params, lr=sampled["lr"], weight_decay=sampled["weight_decay"])

    print(f"config: depth={sampled['cnn_depth']} base_ch={sampled['cnn_base_channels']} "
          f"k={sampled['cnn_kernel']} freq_down={sampled['cnn_freq_downsample']} "
          f"act={sampled['cnn_activation']} enc_hidden={sampled['enc_hidden']} "
          f"layers={sampled['lstm_layers']} head={sampled['head_ratio']} "
          f"lambda={sampled['lambda_reg']} lr={sampled['lr']:.2e} bs={sampled['batch_size']} "
          f"wd={sampled['weight_decay']:.1e} clip={sampled['grad_clip']} opt={sampled['optimizer']}")

    batch_size = sampled["batch_size"]
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    n_train_batches = len(train_loader)

    for epoch in range(epochs):
        encoder_cnn.train(); encoder_lstm.train(); decoder.train()
        train_tot = 0.0
        for b, (source_batch, target_batch) in enumerate(train_loader):
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            decoder_outputs = forward_pass(source_batch, target_batch, encoder_cnn, encoder_lstm, decoder)
            total_loss, _ = compute_masked_loss(decoder_outputs, target_batch, cfg)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=sampled["grad_clip"])
            optimizer.step()
            train_tot += total_loss.item()
            print(f"    epoch {epoch + 1}/{epochs}  batch {b + 1}/{n_train_batches}"
                  f"  train {train_tot / (b + 1):.4f}", end="\r")

        encoder_cnn.eval(); encoder_lstm.eval(); decoder.eval()
        total_val_loss, num_val_batches = 0.0, 0
        with torch.no_grad():
            for source_batch, target_batch in val_loader:
                source_batch = source_batch.to(device)
                target_batch = target_batch.to(device)
                decoder_outputs = forward_pass(source_batch, target_batch, encoder_cnn, encoder_lstm, decoder)
                l, _ = compute_masked_loss(decoder_outputs, target_batch, cfg)
                total_val_loss += l.item()
                num_val_batches += 1

        val_loss = total_val_loss / max(num_val_batches, 1)
        print(f"    epoch {epoch + 1}/{epochs}  train {train_tot / n_train_batches:.4f}"
              f"  val {val_loss:.4f}            ")
        tune.report({"val_loss": val_loss, "epoch": epoch})


def load_data(input_path, target_path, subset, seed, cfg):

    """
    Loads the spectrograms and padded targets, scales the regression targets, isolates the same test
    set used by evaluate.py so it is never seen during tuning, and carves a tuning-validation split
    out of the remaining training data.

    ----------

    Parameters:
        input_path (str) - path to the spectrogram input file (INPUT.npy).
        target_path (str) - path to the padded target file (padded_sequences.npy).
        subset (int) - if positive and smaller than the training set, subsample the training set to this size.
        seed (int) - random seed for the train/test/validation splits.
        cfg (dict) - configuration dictionary; supplies the scaling constants and test fraction.

    Returns:
        data (tuple) - (X_train, y_train, X_val, y_val, input_hw) ready for the trainable.
    """

    X = array_to_tensor(np.load(input_path, allow_pickle=True))
    y = np.load(target_path, allow_pickle=True).astype(np.float32)
    y[:, :, 4:6] /= cfg["time_max"]
    y[:, :, 6:8] /= cfg["freq_max"]

    X_train_full, _X_test, y_train_full, _y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed)

    if subset and subset < len(X_tr):
        X_tr, y_tr = X_tr[:subset], y_tr[:subset]

    y_tr = torch.tensor(y_tr, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    input_hw = (X.shape[2], X.shape[3])
    return X_tr, y_tr, X_val, y_val, input_hw


def main():

    """
    Parses the command-line arguments, loads the data once, and runs the Ray Tune random search with
    optional ASHA early stopping. Prints the best configuration (already translated into the model
    CONFIG schema) and writes all trial results to a CSV file.

    ----------

    Parameters:
        None

    Returns:
        None
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="INPUT.npy")
    ap.add_argument("--target", default="padded_sequences.npy")
    ap.add_argument("--num-samples", type=int, default=40, help="number of random configs")
    ap.add_argument("--epochs", type=int, default=25, help="max epochs per trial")
    ap.add_argument("--grace", type=int, default=5, help="ASHA grace period (epochs)")
    ap.add_argument("--subset", type=int, default=8000, help="subsample training set (0 = all)")
    ap.add_argument("--gpu-per-trial", type=float, default=1.0)
    ap.add_argument("--cpus-per-trial", type=int, default=2)
    ap.add_argument("--max-concurrent", type=int, default=0, help="0 = let Ray decide")
    ap.add_argument("--no-asha", action="store_true", help="disable early stopping")
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--storage", default=os.path.abspath("./ray_results"))
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True)

    data = load_data(args.input, args.target, args.subset, args.seed, CONFIG)
    print(f"Tuning on {len(data[0])} train / {len(data[2])} val samples, input HxW = {data[4]}")

    trainable = tune.with_parameters(
        train_trainable, data=data, epochs=args.epochs, seed=args.seed)
    trainable = tune.with_resources(
        trainable, {"cpu": args.cpus_per_trial, "gpu": args.gpu_per_trial})

    scheduler = None
    if not args.no_asha:
        scheduler = ASHAScheduler(max_t=args.epochs, grace_period=args.grace, reduction_factor=3)

    tuner = tune.Tuner(
        trainable,
        param_space=search_space(),
        tune_config=tune.TuneConfig(
            metric="val_loss", mode="min",
            num_samples=args.num_samples,          # random search (default search algorithm)
            scheduler=scheduler,
            max_concurrent_trials=(args.max_concurrent or None),
        ),
        run_config=tune.RunConfig(name="cnn_lstm_pulse_search", storage_path=args.storage),
    )
    results = tuner.fit()

    best = results.get_best_result(metric="val_loss", mode="min")
    print("\n" + "=" * 70 + "\nBEST CONFIG\n" + "=" * 70)
    for k, v in best.config.items():
        print(f"  {k:20s} : {v}")
    print(f"\n  best val_loss : {best.metrics['val_loss']:.4f}\n" + "=" * 70)
    print("Best model CONFIG (ready to paste into model.CONFIG):")
    print(to_model_cfg(best.config))

    results.get_dataframe().to_csv("tuning_results.csv", index=False)
    print("Saved full results to tuning_results.csv")


if __name__ == "__main__":
    main()
