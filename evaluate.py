import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model import build_models, concatenate_bidirectional_states, CONFIG
from functions import array_to_tensor
from metrics_core import evaluate
from plots import make_all_plots



cfg = CONFIG
seed = cfg["seed"]
np.random.seed(seed)
torch.manual_seed(seed)

time_max = cfg["time_max"]
freq_max = cfg["freq_max"]
class_names = {0: "cw", 1: "lfm", 2: "hfm", 3: "eos"}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the checkpoint to evaluate here.
CHECKPOINT = "best_model_epoch_20_valloss_0.3393.pth"


@torch.no_grad()
def predict(source):

    """
    Runs autoregressive inference on a batch of spectrograms.

    ----------

    Parameters:
        source (torch.float32) - batch of spectrograms of shape [batch, 1, frequency_bins, time_bins].

    Returns:
        results (list) - one list of pulse dicts per spectrogram, with keys type, t_start, t_stop, f1, f2.
    """

    source = source.to(device)
    feats = encoder_cnn(source)
    _, (h, c) = encoder_lstm(feats)
    dh, dc = concatenate_bidirectional_states(h, c)
    out, _ = decoder.generate(dh, dc, max_length=cfg["max_length"])

    batch = source.shape[0]
    results = []
    for b in range(batch):
        cls = out['classification'][b]
        pulses = []
        for s in range(cls.shape[0]):
            k = int(torch.argmax(cls[s]))
            if k == cfg["eos_token_id"]:
                break
            pulses.append({
                "type":    class_names[k],
                "t_start": float(out['start_time'][b, s, 0]) * time_max,
                "t_stop":  float(out['end_time'][b, s, 0]) * time_max,
                "f1":      float(out['start_freq'][b, s, 0]) * freq_max,
                "f2":      float(out['end_freq'][b, s, 0]) * freq_max,
            })
        results.append(pulses)
    return results


def ground_truth(target):

    """
    Reads the true pulses for one spectrogram from its target tensor, stopping at the first EOS.

    ----------

    Parameters:
        target (torch.float32) - target tensor of shape [seq, 8], rows
                                 [cw, lfm, hfm, eos, t_start, t_stop, f1, f2].

    Returns:
        pulses (list) - list of pulse dicts with keys type, t_start, t_stop, f1, f2.
    """

    pulses = []
    for s in range(target.shape[0]):
        k = int(torch.argmax(target[s, :4]))
        if k == cfg["eos_token_id"]:
            break
        pulses.append({
            "type":    class_names[k],
            "t_start": float(target[s, 4]),
            "t_stop":  float(target[s, 5]),
            "f1":      float(target[s, 6]),
            "f2":      float(target[s, 7]),
        })
    return pulses


def fmt(pulses):

    """
    Formats a list of pulses as text for printing, one pulse per line.

    ----------

    Parameters:
        pulses (list) - list of pulse dicts with keys type, t_start, t_stop, f1, f2.

    Returns:
        (str) - formatted block, or '(none)' when the list is empty.
    """

    if not pulses:
        return "    (none)"
    return "\n".join(
        f"    {p['type']:4s} t=[{p['t_start']:.2f}, {p['t_stop']:.2f}] s "
        f"f=[{p['f1']:.0f}, {p['f2']:.0f}] Hz"
        for p in pulses
    )


def print_report(m):

    """
    Prints the aggregate metrics returned by metrics_core.evaluate, covering pulse count, detection,
    classification, per-quantity regression error, and a plus/minus summary based on the median and
    90th-percentile absolute error.

    ----------

    Parameters:
        m (dict) - metrics dictionary returned by metrics_core.evaluate.

    Returns:
        None
    """

    print("\n" + "=" * 70)
    print(f"AGGREGATE METRICS  ({m['n_samples']} test samples)")
    print("=" * 70)

    c = m["count"]
    print("\n--- Pulse COUNT ---")
    print(f"  exact count match : {c['exact_match_rate']*100:5.1f} %   "
          f"(predicted #pulses == true #pulses)")
    print(f"  within +/-1 pulse : {c['within_1']*100:5.1f} %")
    print(f"  count bias        : {c['count_bias']:+.3f}  "
          f"(>0 over-predicts, <0 under-predicts)")
    print(f"  count MAE         : {c['count_mae']:.3f} pulses")

    d = m["detection"]
    print("\n--- DETECTION (pulse-level, after matching) ---")
    print(f"  TP={d['TP']}  FP={d['FP']} (false alarms)  FN={d['FN']} (misses)")
    print(f"  precision={d['precision']:.3f}  recall={d['recall']:.3f}  f1={d['f1']:.3f}")

    a = m["classification_on_matched"]
    print("\n--- CLASSIFICATION (on matched pulses) ---")
    print(f"  type accuracy     : {a['accuracy']*100:5.1f} %   (n={a['n']})")

    print("\n--- REGRESSION error on matched pulses ---")
    print(f"  {'param':8s} {'MAE':>10s} {'RMSE':>10s} {'bias':>10s} {'p90':>10s}")
    for key in ["t_start", "t_stop", "f1", "f2"]:
        s = m["regression"][key]
        if s is None:
            print(f"  {key:8s}   (no matched pulses)")
            continue
        u = "s" if key.startswith("t") else "Hz"
        print(f"  {key:8s} {s['mae']:9.4f}{u:>1s} {s['rmse']:9.4f}{u:>1s} "
              f"{s['bias']:+9.4f}{u:>1s} {s['p90']:9.4f}{u:>1s}")

    ts, fs = m["regression"]["t_start"], m["regression"]["f1"]
    if ts and fs:
        print("\n--- Headline +/- (matched pulses) ---")
        print(f"  time  : typical +/-{ts['p50']:.3f} s  (90% within +/-{ts['p90']:.3f} s)")
        print(f"  freq  : typical +/-{fs['p50']:.0f} Hz (90% within +/-{fs['p90']:.0f} Hz)")
    print("=" * 70)


X = np.load("INPUT.npy", allow_pickle=True)
X = array_to_tensor(X)

y = np.load("padded_sequences.npy", allow_pickle=True)

_, X_test, _, y_test = train_test_split(X, y, test_size=cfg["test_size"], random_state=seed)
y_test = torch.tensor(y_test, dtype=torch.float32)
print(f"Test set: {X_test.shape[0]} samples")

# Build the same architecture as training, from CONFIG, then load the weights.
input_hw = (X.shape[2], X.shape[3])
encoder_cnn, encoder_lstm, decoder = build_models(input_hw, cfg, device)

ckpt = torch.load(CHECKPOINT, map_location=device)
encoder_cnn.load_state_dict(ckpt['encoder_cnn_state_dict'])
encoder_lstm.load_state_dict(ckpt['encoder_lstm_state_dict'])
decoder.load_state_dict(ckpt['decoder_state_dict'])
print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_loss {ckpt['validation_loss']:.4f}")

encoder_cnn.eval()
encoder_lstm.eval()
decoder.eval()

batch_size = 6
all_pred, all_truth = [], []
for start in range(0, X_test.shape[0], batch_size):
    xb = X_test[start:start + batch_size]
    yb = y_test[start:start + batch_size]
    preds = predict(xb)
    for i, pred in enumerate(preds):
        idx = start + i
        truth = ground_truth(yb[i])
        all_pred.append(pred)
        all_truth.append(truth)
        print(f"\n=== test sample {idx} ===")
        print("  PREDICTED:")
        print(fmt(pred))
        print("  GROUND TRUTH:")
        print(fmt(truth))

#metrics = evaluate(all_pred, all_truth)
#print_report(metrics)
#make_all_plots(all_pred, all_truth, metrics, outdir="eval_plots", prefix="eval")
