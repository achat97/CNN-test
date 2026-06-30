import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import build_models, forward_pass, compute_masked_loss, CONFIG
from functions import array_to_tensor, save_padded_sequences



cfg = CONFIG
seed = cfg["seed"]
np.random.seed(seed)
torch.manual_seed(seed)

X = np.load("INPUT.npy", allow_pickle=True)
X = array_to_tensor(X)

#y = np.load("TARGET.npy", allow_pickle=True)
#y_padded = save_padded_sequences(y, filename='padded_sequences.npy', target_length=cfg["max_length"])
y = np.load("padded_sequences.npy", allow_pickle=True).astype(np.float32)

# scale target data using the constants in CONFIG (were the literals 5 and 15000)
y[:, :, 4:6] = y[:, :, 4:6] / cfg["time_max"]
y[:, :, 6:8] = y[:, :, 6:8] / cfg["freq_max"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg["test_size"], random_state=seed)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device, "| cuda available:", torch.cuda.is_available())

# Build all three modules from CONFIG. The formerly-hardcoded 512*15 and 1024 are
# derived inside build_models from the input shape and enc_hidden.
input_hw = (X.shape[2], X.shape[3])
encoder_cnn, encoder_lstm, decoder = build_models(input_hw, cfg, device)

optimizer = torch.optim.Adam(
    list(encoder_cnn.parameters()) +
    list(encoder_lstm.parameters()) +
    list(decoder.parameters()),
    lr=0.001)

num_epochs = 50
batch_size = 32

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train_model(num_epochs):

    """
    Trains the encoder-decoder model over a specified number of epochs, tracking training and
    validation losses across epochs. For each epoch, the model is run in training mode over the
    training set, computing the combined masked loss for each batch and updating parameters using Adam.
    Component losses (classification, start/end frequency, start/end time) are accumulated and
    averaged across batches for logging. After every epoch the model is evaluated on the validation
    set and the best validation loss is tracked; the model weights corresponding to the best
    validation loss are saved to disk via save_best_model.

    ----------

    Parameters:
        num_epochs (int) - number of training epochs.

    Returns:
        train_losses (list) - list of average training losses per epoch.
        val_losses (list) - list of validation losses (one per epoch).
    """

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        total_epoch_loss = 0
        num_batches = 0

        epoch_classification_loss = 0
        epoch_start_freq_loss = 0
        epoch_end_freq_loss = 0
        epoch_start_time_loss = 0
        epoch_end_time_loss = 0

        encoder_cnn.train()
        encoder_lstm.train()
        decoder.train()

        for batch_idx, (source_batch, target_batch) in enumerate(train_loader):
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)

            decoder_outputs = forward_pass(source_batch, target_batch, encoder_cnn, encoder_lstm, decoder)

            total_loss, component_losses = compute_masked_loss(decoder_outputs, target_batch, cfg)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder_cnn.parameters())
                + list(encoder_lstm.parameters())
                + list(decoder.parameters()),
                max_norm=1.0)
            optimizer.step()

            total_epoch_loss += total_loss.item()
            epoch_classification_loss += component_losses['classification'].item()
            epoch_start_freq_loss += component_losses['start_freq'].item()
            epoch_end_freq_loss += component_losses['end_freq'].item()
            epoch_start_time_loss += component_losses['start_time'].item()
            epoch_end_time_loss += component_losses['end_time'].item()
            num_batches += 1

            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}")
            avg_class = epoch_classification_loss / num_batches
            avg_freq = (epoch_start_freq_loss + epoch_end_freq_loss) / (2 * num_batches)
            avg_time = (epoch_start_time_loss + epoch_end_time_loss) / (2 * num_batches)

            print(f"  Components - Class: {avg_class:.4f}, Freq: {avg_freq:.4f}, Time: {avg_time:.4f}")

        avg_epoch_loss = total_epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")
        print("-------------------------------------------------------------------------------")

        val_loss = validate_model()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            print(f"New best validation loss: {val_loss:.4f} at epoch {epoch}")

            save_best_model(epoch, val_loss)

    print(f"Training completed! Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    np.save("train_losses.npy", np.array(train_losses))
    np.save("val_losses.npy", np.array(val_losses))

    print("Saved train_losses.npy and val_losses.npy")

    return train_losses, val_losses

def validate_model():

    """
    Evaluates the model on the validation set without updating parameters. Runs under torch.no_grad
    in evaluation mode, computes the masked loss for each batch, and returns the average across
    batches, then restores training mode.

    ----------

    Parameters:
        None

    Returns:
        avg_val_loss (float) - average validation loss across all batches in test_loader.
    """

    encoder_cnn.eval()
    encoder_lstm.eval()
    decoder.eval()

    total_val_loss = 0
    num_val_batches = 0

    with torch.no_grad():
        for source_batch, target_batch in test_loader:
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            decoder_outputs = forward_pass(source_batch, target_batch, encoder_cnn, encoder_lstm, decoder)
            total_loss, component_losses = compute_masked_loss(decoder_outputs, target_batch, cfg)
            total_val_loss += total_loss.item()
            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches
    print(f"Validation Loss: {avg_val_loss:.4f}")

    encoder_cnn.train()
    encoder_lstm.train()
    decoder.train()

    return avg_val_loss


def save_best_model(epoch, val_loss):

    """
    Saves the model weights when validation loss is lowest. Saves a checkpoint containing the epoch,
    validation loss, the configuration used, and state dictionaries of the CNN encoder, LSTM encoder,
    decoder, and optimizer to a uniquely named .pth file.

    ----------

    Parameters:
        epoch (int) - epoch at which the best validation loss was achieved.
        val_loss (float) - validation loss value used in the saved filename.

    Returns:
        None
    """

    checkpoint = {
        'epoch': epoch,
        'validation_loss': val_loss,
        'config': cfg,
        'encoder_cnn_state_dict': encoder_cnn.state_dict(),
        'encoder_lstm_state_dict': encoder_lstm.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    filename = f'best_model_epoch_{epoch}_valloss_{val_loss:.4f}.pth'
    torch.save(checkpoint, filename)
    print(f"Best model saved as: {filename}")


# Start training!
#print("Starting training...")
#train_losses, val_losses = train_model(num_epochs=num_epochs)
#print("Training completed!")
