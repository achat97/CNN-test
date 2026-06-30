import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import ReshapeForLSTM



# --------------------------------------------------------------------------- #
#  Single source of truth: every value that used to be hardcoded lives here.
#  Change a value here and both build_models() and the tuning script follow it.
# --------------------------------------------------------------------------- #
CONFIG = {
    # CNN encoder
    "cnn_channels":     [16, 32, 64, 128, 256, 512],  # one entry per conv block
    "cnn_kernel":       (3, 3),
    "cnn_padding":      (0, 1),
    "cnn_stride":       (2, 1),   # a single (freq, time) tuple applied to every block, OR a
                                  # list of per-block tuples of length len(cnn_channels). The
                                  # frequency stride per block controls how aggressively the
                                  # frequency axis is downsampled.
    "cnn_dropout":      0.1,
    "cnn_activation":   "elu",    # one of: elu, gelu, relu, leaky_relu
    # LSTM encoder
    "enc_hidden":       512,
    "lstm_layers":      2,        # shared by encoder and decoder (see build_models)
    "enc_lstm_dropout": 0.0,      # only active when lstm_layers > 1
    "dec_lstm_dropout": 0.0,      # only active when lstm_layers > 1
    # decoder
    "token_size":       8,        # length of one target token
    "head_ratio":       0.25,     # head bottleneck width = hidden_size * head_ratio (was hidden/4)
    "num_classes":      4,        # cw, lfm, hfm, eos
    "eos_token_id":     3,
    "max_length":       10,
    # loss
    "lambda_reg":       10.0,     # weight on the summed regression MSE (was the literal 10.0)
    # target scaling (regression targets are divided by these before training)
    "time_max":         5.0,      # seconds (was the literal 5)
    "freq_max":         15000.0,  # Hz      (was the literal 15000)
    # data split
    "test_size":        0.33,
    "seed":             0,
}

_ACT = {"relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}


def strides_per_block(cfg):

    """
    Expands the stride entry of a configuration into one (frequency, time) stride tuple per
    convolutional block. The stride may be given either as a single tuple, in which case it is
    applied to every block, or as a list of per-block tuples, which allows the frequency axis to be
    downsampled by only some of the blocks.

    ----------

    Parameters:
        cfg (dict) - configuration dictionary containing 'cnn_channels' and 'cnn_stride'.

    Returns:
        strides (list) - list of (frequency_stride, time_stride) tuples, one per block.
    """

    n = len(cfg["cnn_channels"])
    stride = cfg["cnn_stride"]

    if isinstance(stride[0], (tuple, list)):
        if len(stride) != n:
            raise ValueError("cnn_stride list must have one entry per conv block")
        return [tuple(s) for s in stride]

    return [tuple(stride)] * n


class CNNEncoder(nn.Module):

    """
    Initializes CNN architecture and reshapes the output in preparation for LSTM encoding. The number
    of blocks, their channels, kernel, padding, stride, dropout, and activation are all read from the
    configuration dictionary, so the architecture can be changed without editing the class. The stride
    may be a single (frequency, time) tuple applied to every block or a list of per-block tuples, which
    lets the frequency axis be downsampled by only some of the blocks.
    """

    def __init__(self, cfg=CONFIG):
        super().__init__()
        act = _ACT[cfg.get("cnn_activation", "elu")]
        strides = strides_per_block(cfg)

        blocks, in_ch = [], 1
        for out_ch, stride in zip(cfg["cnn_channels"], strides):
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=cfg["cnn_kernel"],
                          padding=cfg["cnn_padding"], stride=stride),
                nn.BatchNorm2d(out_ch),
                act(),
                nn.Dropout2d(cfg["cnn_dropout"]),
            ]
            in_ch = out_ch

        self.cnn = nn.Sequential(*blocks)
        self.reshape = ReshapeForLSTM()

    def forward(self, x):

        """
        Forward pass through the CNN followed by a reshape of the feature maps into a sequence.

        ----------

        Parameters:
            x (torch.float32) - Spectrogram(s) of shape [batch_size, 1, height, width].

        Returns:
            x (torch.float32) - Concatenated feature maps of shape [batch_size, sequence_length, hidden_size].
        """

        x = self.cnn(x)
        x = self.reshape(x)
        return x


class LSTMDecoder(nn.Module):

    """
    Initializes the decoder consisting of an LSTM layer followed by fully connected output layers.
    The hidden size, number of layers, dropout, head bottleneck ratio, token size, and number of
    classes are all read from the configuration dictionary.
    """

    def __init__(self, hidden_size, cfg=CONFIG):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = cfg["lstm_layers"]
        self.cfg = cfg

        self.lstm = nn.LSTM(cfg["token_size"], hidden_size, cfg["lstm_layers"],
                            batch_first=True, bidirectional=False,
                            dropout=cfg.get("dec_lstm_dropout", 0.0) if cfg["lstm_layers"] > 1 else 0.0)

        head_hidden = max(1, int(hidden_size * cfg["head_ratio"]))

        def make_head(out_dim):
            return nn.Sequential(
                nn.Linear(hidden_size, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, out_dim),
            )

        self.classification_head = make_head(cfg["num_classes"])
        self.start_freq_head = make_head(1)
        self.end_freq_head = make_head(1)
        self.start_time_head = make_head(1)
        self.end_time_head = make_head(1)

    def forward(self, target_sequence, encoder_hidden, encoder_cell, max_length=None, eos_token_id=None):

        """
        Teacher-forced forward pass through the decoder. At each step it is fed the ground-truth token
        from the previous position (step 0 uses a start token) and predicts the token at the current
        position. Runs the full max_length steps; padding and EOS are handled later by the masked loss.

        ----------

        Parameters:
            target_sequence (torch.float32) - target data of shape [batch_size, sequence_length, 8].
            encoder_hidden (torch.float32) - final encoder hidden state, [num_layers, batch_size, hidden_size].
            encoder_cell (torch.float32) - final encoder cell state, [num_layers, batch_size, hidden_size].
            max_length (int) - number of output steps to generate; defaults to cfg['max_length'].
            eos_token_id (int) - index of the EOS flag within the 8-element token; defaults to cfg['eos_token_id'].

        Returns:
            final_outputs (dict) - stacked predictions per head, each [batch_size, max_length, ...].
            finished (torch.bool) - per-sample flag, whether an EOS was predicted.
        """

        max_length = self.cfg["max_length"] if max_length is None else max_length
        eos_token_id = self.cfg["eos_token_id"] if eos_token_id is None else eos_token_id

        batch_size = encoder_hidden.size(1)
        device = encoder_hidden.device

        current_input = torch.zeros(batch_size, 1, self.cfg["token_size"], device=device)
        current_input[:, :, eos_token_id] = 1.0

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        all_outputs = {
            'classification': [],
            'start_freq': [],
            'end_freq': [],
            'start_time': [],
            'end_time': []
        }

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length):

            if step >= target_sequence.size(1):
                break

            decoder_output, (decoder_hidden, decoder_cell) = self.lstm(
                current_input, (decoder_hidden, decoder_cell)
            )

            decoder_output_squeezed = decoder_output.squeeze(1)

            classification_output = self.classification_head(decoder_output_squeezed)
            start_freq_output = self.start_freq_head(decoder_output_squeezed)
            end_freq_output = self.end_freq_head(decoder_output_squeezed)
            start_time_output = self.start_time_head(decoder_output_squeezed)
            end_time_output = self.end_time_head(decoder_output_squeezed)

            all_outputs['classification'].append(classification_output)
            all_outputs['start_freq'].append(start_freq_output)
            all_outputs['end_freq'].append(end_freq_output)
            all_outputs['start_time'].append(start_time_output)
            all_outputs['end_time'].append(end_time_output)

            predicted_classes = torch.argmax(classification_output, dim=-1)
            is_eos = (predicted_classes == eos_token_id)
            finished = finished | is_eos

            current_input = target_sequence[:, step:step + 1, :]

        final_outputs = {}
        for key in all_outputs:
            final_outputs[key] = torch.stack(all_outputs[key], dim=1)

        return final_outputs, finished

    @torch.no_grad()
    def generate(self, encoder_hidden, encoder_cell, max_length=None, eos_token_id=None):

        """
        Autoregressive inference. The decoder feeds its own prediction back in as the next input,
        forming each next input as an 8-element token [cw, lfm, hfm, eos, t_start, t_stop, f1, f2] from the head outputs.
        Stops once every sample has predicted an EOS, or after max_length steps.

        ----------

        Parameters:
            encoder_hidden (torch.float32) - final encoder hidden state, [num_layers, batch_size, hidden_size].
            encoder_cell (torch.float32) - final encoder cell state, [num_layers, batch_size, hidden_size].
            max_length (int) - largest number of steps to generate; defaults to cfg['max_length'].
            eos_token_id (int) - index of the EOS flag within the 8-element token; defaults to cfg['eos_token_id'].

        Returns:
            (dict) - the predictions at every step, one entry per output: the class scores of shape
                    [batch_size, steps, 4], and the four time/frequency values, each [batch_size, steps, 1],
                    where steps is the number generated (at most max_length).
            finished (torch.bool) - one flag per sample, True if it predicted an EOS.
        """

        max_length = self.cfg["max_length"] if max_length is None else max_length
        eos_token_id = self.cfg["eos_token_id"] if eos_token_id is None else eos_token_id
        n_cls = self.cfg["num_classes"]
        token_size = self.cfg["token_size"]

        batch_size = encoder_hidden.size(1)
        device = encoder_hidden.device

        current_input = torch.zeros(batch_size, 1, token_size, device=device)
        current_input[:, :, eos_token_id] = 1.0  # start token

        h, c = encoder_hidden, encoder_cell
        outputs = {k: [] for k in
                   ['classification', 'start_freq', 'end_freq', 'start_time', 'end_time']}
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            out, (h, c) = self.lstm(current_input, (h, c))
            out = out.squeeze(1)

            cls = self.classification_head(out)
            sf, ef = self.start_freq_head(out), self.end_freq_head(out)
            st, et = self.start_time_head(out), self.end_time_head(out)

            outputs['classification'].append(cls)
            outputs['start_freq'].append(sf)
            outputs['end_freq'].append(ef)
            outputs['start_time'].append(st)
            outputs['end_time'].append(et)

            pred_cls = torch.argmax(cls, dim=-1)
            finished = finished | (pred_cls == eos_token_id)
            if finished.all():
                break

            # build the next token
            nxt = torch.zeros(batch_size, 1, token_size, device=device)
            nxt[:, 0, :n_cls] = F.one_hot(pred_cls, num_classes=n_cls).float()
            nxt[:, 0, 4] = st.squeeze(-1)   # t_start
            nxt[:, 0, 5] = et.squeeze(-1)   # t_stop
            nxt[:, 0, 6] = sf.squeeze(-1)   # f1
            nxt[:, 0, 7] = ef.squeeze(-1)   # f2
            current_input = nxt

        return {k: torch.stack(v, dim=1) for k, v in outputs.items()}, finished


def concatenate_bidirectional_states(hidden, cell):

    """
    Concatenates the forward and backward states of a bidirectional LSTM into a single set suitable for initializing a unidirectional decoder.

    ----------

    Parameters:
        hidden (torch.float32) - hidden state of the bidirectional LSTM with shape (num_layers*2, batch, hidden_size).
        cell (torch.float32) - cell state of the bidirectional LSTM with shape (num_layers*2, batch, hidden_size).

    Returns:
        decoder_hidden (torch.float32) - concatenated hidden state with shape (num_layers, batch, 2*hidden_size).
        decoder_cell (torch.float32) - concatenated cell state with shape (num_layers, batch, 2*hidden_size).
    """

    forward_hidden = hidden[::2]
    backward_hidden = hidden[1::2]

    forward_cell = cell[::2]
    backward_cell = cell[1::2]

    decoder_hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)
    decoder_cell = torch.cat([forward_cell, backward_cell], dim=-1)

    return decoder_hidden, decoder_cell


def build_models(input_hw, cfg=CONFIG, device="cpu"):

    """
    Builds the full encoder-decoder pipeline (CNN encoder, bidirectional LSTM encoder, LSTM decoder)
    from a configuration dictionary. The two quantities that used to be hardcoded are derived here
    rather than written by hand: the encoder-LSTM input size is the product of the last CNN channel
    count and the final frequency height, obtained from a dummy forward pass through the CNN, and the
    decoder hidden size is twice the encoder hidden size, since the forward and backward encoder states
    are concatenated. The decoder uses the same number of layers as the encoder so that the
    concatenated states can be used to initialize it.

    ----------

    Parameters:
        input_hw (tuple) - the (height, width) of a single input spectrogram, i.e. (frequency_bins, time_bins).
        cfg (dict) - configuration dictionary; defaults to CONFIG.
        device (str or torch.device) - device on which the modules are created.

    Returns:
        encoder_cnn (CNNEncoder) - the convolutional encoder.
        encoder_lstm (nn.LSTM) - the bidirectional LSTM encoder.
        decoder (LSTMDecoder) - the LSTM decoder with classification and regression heads.
    """

    height, width = input_hw
    encoder_cnn = CNNEncoder(cfg).to(device)

    with torch.no_grad():
        dummy = torch.zeros(1, 1, height, width, device=device)
        flatten_dim = encoder_cnn(dummy).shape[-1]

    encoder_lstm = nn.LSTM(
        input_size=flatten_dim, hidden_size=cfg["enc_hidden"],
        num_layers=cfg["lstm_layers"], batch_first=True, bidirectional=True,
        dropout=cfg.get("enc_lstm_dropout", 0.0) if cfg["lstm_layers"] > 1 else 0.0,
    ).to(device)

    decoder = LSTMDecoder(hidden_size=2 * cfg["enc_hidden"], cfg=cfg).to(device)

    return encoder_cnn, encoder_lstm, decoder


def forward_pass(source_data, target_data, encoder_cnn, encoder_lstm, decoder):

    """
    Complete forward pass through the encoder-decoder pipeline. Source data is passed through the CNN
    encoder to extract feature maps, which are then processed by a bidirectional LSTM encoder. The
    bidirectional encoder states are concatenated to match the decoder's expected input shape, and
    the decoder produces output predictions using the target data with teacher forcing.

    ----------

    Parameters:
        source_data (torch.float32) - input spectrograms with shape [batch_size, 1, height, width].
        target_data (torch.float32) - target sequence with shape [batch_size, sequence_length, 8].
        encoder_cnn (nn.Module) - CNN encoder module.
        encoder_lstm (nn.Module) - bidirectional LSTM encoder module.
        decoder (nn.Module) - LSTM decoder module with output heads.

    Returns:
        decoder_outputs (dict) - dictionary containing the decoder's classification and regression outputs.
    """

    # 1. Encode through CNN
    cnn_features = encoder_cnn(source_data)

    # 2. Encode through bidirectional LSTM
    encoder_output, (encoder_hidden, encoder_cell) = encoder_lstm(cnn_features)

    # 3. Concatenate bidirectional states for decoder
    decoder_hidden, decoder_cell = concatenate_bidirectional_states(encoder_hidden, encoder_cell)

    # 4. Decode - note: decoder expects (target_seq, hidden, cell)
    decoder_outputs, _ = decoder(target_data, decoder_hidden, decoder_cell)

    return decoder_outputs


def compute_masked_loss(predictions, targets, cfg=CONFIG):

    """
    Computes the combined masked loss for the encoder-decoder model.

    Classification (cross-entropy) is scored on every position up to and including the first EOS.
    The four regression outputs (start/end time and frequency) are scored only on the real pulse
    positions, since the EOS row has zero time/frequency values. Each component is masked and
    averaged over its valid positions, then combined into one weighted total using cfg['lambda_reg'].

    ----------

    Parameters:
        predictions (dict) - model outputs with keys 'classification', 'start_freq', 'end_freq',
                            'start_time', 'end_time'.
        targets (torch.float32) - target tensor of shape [batch_size, sequence_length, 8].
        cfg (dict) - configuration dictionary; supplies 'lambda_reg' and 'num_classes'.

    Returns:
        total_loss (torch.float32) - weighted sum of the loss components.
        loss_dict (dict) - the individual unweighted components, for logging.
    """

    batch_size, seq_len, _ = targets.shape

    target_classes = targets[:, :, 3]
    eos_flag = (target_classes > 0.5)

    class_mask = torch.zeros_like(eos_flag, dtype=torch.bool)
    for i in range(batch_size):
        first_eos = torch.where(eos_flag[i])[0][0]
        class_mask[i, :first_eos + 1] = True

    reg_mask = class_mask & (~eos_flag)

    pred_seq_len = predictions['classification'].size(1)

    class_idx    = torch.argmax(targets[:, :pred_seq_len, :4], dim=-1)
    start_time_t = targets[:, :pred_seq_len, 4]
    end_time_t   = targets[:, :pred_seq_len, 5]
    start_freq_t = targets[:, :pred_seq_len, 6]
    end_freq_t   = targets[:, :pred_seq_len, 7]

    class_mask = class_mask[:, :pred_seq_len].float()
    reg_mask   = reg_mask[:, :pred_seq_len].float()
    reg_denom  = reg_mask.sum().clamp(min=1.0)

    classification_loss = F.cross_entropy(
        predictions['classification'].reshape(-1, cfg["num_classes"]),
        class_idx.reshape(-1),
        reduction='none'
    ).reshape(batch_size, pred_seq_len)

    start_freq_loss = F.mse_loss(predictions['start_freq'].squeeze(-1), start_freq_t, reduction='none')
    end_freq_loss   = F.mse_loss(predictions['end_freq'].squeeze(-1),   end_freq_t,   reduction='none')
    start_time_loss = F.mse_loss(predictions['start_time'].squeeze(-1), start_time_t, reduction='none')
    end_time_loss   = F.mse_loss(predictions['end_time'].squeeze(-1),   end_time_t,   reduction='none')

    classification_loss = (classification_loss * class_mask).sum() / class_mask.sum().clamp(min=1.0)
    start_freq_loss = (start_freq_loss * reg_mask).sum() / reg_denom
    end_freq_loss   = (end_freq_loss   * reg_mask).sum() / reg_denom
    start_time_loss = (start_time_loss * reg_mask).sum() / reg_denom
    end_time_loss   = (end_time_loss   * reg_mask).sum() / reg_denom

    reg_loss = start_freq_loss + end_freq_loss + start_time_loss + end_time_loss
    total_loss = classification_loss + cfg["lambda_reg"] * reg_loss

    return total_loss, {
        'classification': classification_loss,
        'start_freq': start_freq_loss,
        'end_freq': end_freq_loss,
        'start_time': start_time_loss,
        'end_time': end_time_loss
    }
