# Installation and Imports
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load and clean the dataset
excel_path = "C:/Users/Benyamin/Documents/KAN/FinalDataNN.xlsx"
df = pd.read_excel(excel_path)

# Drop rows containing NaN in the 4 key columns
df_cleaned = df.dropna(subset=["P", "Y", "Ptest", "Ytest"])

# Convert to numpy arrays (float32)
P_train = df_cleaned["P"].values.astype("float32")
Y_train = df_cleaned["Y"].values.astype("float32")
P_test  = df_cleaned["Ptest"].values.astype("float32")
Y_test  = df_cleaned["Ytest"].values.astype("float32")

print(f"Train set size:  {len(P_train)} samples")
print(f"Test set size:   {len(P_test)} samples")

# Hyper‑parameters
K_S = 5      # KAN complexity (k)
G_S = 3      # internal hidden dimension of each ReLU-KAN (g)
Batch_Size = 64
Context_Len = 150   # window length for context
Pred_Len = 50       # window length to predict
H_Dims = 32         # dimensionality of hidden layers (KAN output & LSTM hidden)
N_Epochs = 5000

# Displaying Time Series
# Use 'P' and 'Y' columns for plotting
time_series_data_P = df['P']
time_series_data_Y = df['Y']
# Plot the time series data for 'P' and 'Y'
plt.figure(figsize=(12, 6))
plt.plot(time_series_data_P, label='P')
plt.plot(time_series_data_Y, label='Y')
plt.title('Time Series Data (P and Y)')
plt.xlabel('Time (Index)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Sliding‑window dataset (P → Y)
class PairedTimeSeriesDataset(Dataset):
    def __init__(self, x_series, y_series, context_len=Context_Len, pred_len=Pred_Len):
        self.x = torch.tensor(x_series, dtype=torch.float32)
        self.y = torch.tensor(y_series, dtype=torch.float32)
        self.context_len = context_len
        self.pred_len = pred_len
        self.total_len = context_len + pred_len
        self.n_samples = max(len(self.x) - self.total_len + 1, 0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_win = self.x[idx : idx + self.context_len]                 # P‑window
        y_win = self.y[idx + self.context_len : idx + self.total_len]  # corresponding Y‑window
        return x_win, y_win

train_dataset = PairedTimeSeriesDataset(P_train, Y_train)
test_dataset  = PairedTimeSeriesDataset(P_test,  Y_test)

train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=Batch_Size, shuffle=False)

print(f"Train windows: {len(train_dataset)}")
print(f"Test  windows: {len(test_dataset)}")

# ReLU‑KAN layer and the Recurrent model
class ReLUKANLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 g: int,
                 k: int,
                 output_size: int,
                 imin: float,
                 imax: float,
                 train_ab: bool = True):
        """
        ReLU-KAN layer as in the reference implementation:
        - Builds phase_low and phase_height
        - Uses ReLU(x - phase_low) * ReLU(phase_height - x) * r
        - Squares the output and applies a Conv2d to mix (g+k) grid outputs.
        """
        super().__init__()

        self.g = g
        self.k = k
        self.input_size = input_size
        self.output_size = output_size

        # Normalization factor r
        self.r = 4 * g * g / ((k + 1) * (k + 1) * (imax - imin) * (imax - imin))

        # Build phase grid (same logic as your reference ReLUKANLayer)
        phase_low = (imax - imin) * (np.arange(-k, g) / g) - (-imin)
        phase_height = phase_low + (k + 1) / g * (imax - imin)

        # Repeat for each input dimension: shape (input_size, g+k)
        self.phase_low = nn.Parameter(
            torch.tensor(np.array([phase_low for _ in range(input_size)]), dtype=torch.float32),
            requires_grad=train_ab
        )
        self.phase_height = nn.Parameter(
            torch.tensor(np.array([phase_height for _ in range(input_size)]), dtype=torch.float32),
            requires_grad=train_ab
        )

        # Conv2d to map (g+k, input_size) → output_size
        # in_channels = 1, out_channels = output_size
        self.equal_size_conv = nn.Conv2d(
            in_channels=1,
            out_channels=output_size,
            kernel_size=(g + k, input_size)
        )

    def forward(self, x):
        """
        x: (batch, input_size)
        output: (batch, output_size)
        """
        # Expand x to match phase_low shape
        # phase_low shape: (input_size, g+k)  -> treat as (D, G)
        # For broadcasting: x_expanded: (batch, input_size, g+k)
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))

        # Apply ReLU basis
        x1 = torch.relu(x_expanded - self.phase_low)      # (batch, D, G)
        x2 = torch.relu(self.phase_height - x_expanded)   # (batch, D, G)

        # Multiply and normalize
        x = x1 * x2 * self.r
        x = x * x  # square

        # Reshape to (batch, 1, g+k, input_size) for Conv2d
        x = x.reshape(x.size(0), 1, self.g + self.k, self.input_size)

        # Conv2d: (batch, output_size, 1, 1)
        x = self.equal_size_conv(x)

        # Flatten to (batch, output_size)
        x = x.view(x.size(0), self.output_size)
        return x

# Recurrent ReLU-KAN + LSTM model
class RecurrentReLUKANTS(nn.Module):
    """
    Context → ReLU-KAN (piecewise) → LSTM → Linear projection
    Uses the ReLUKANLayer defined above on each chunk of the context window.
    """
    def __init__(self,
                 context_len: int,
                 pred_len: int,
                 g: int,
                 k: int,
                 kan_out_dim: int,
                 rnn_hidden: int,
                 seq_step: int = 10,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 imin: float = 0.0,
                 imax: float = 1.0):
        super().__init__()
        assert context_len % seq_step == 0, "context_len must be a multiple of seq_step"
        self.context_len = context_len
        self.pred_len = pred_len
        self.seq_step = seq_step
        self.num_steps = context_len // seq_step

        # ---- ReLU-KAN ----
        # Each segment with length seq_step goes into ReLUKANLayer as input_size
        self.kan = ReLUKANLayer(
            input_size=seq_step,
            g=g,
            k=k,
            output_size=kan_out_dim,
            imin=imin,
            imax=imax,
            train_ab=True  # If you want the phases to be constant, you can set it to False.
        )

        # ---- LSTM ----
        self.lstm = nn.LSTM(
            input_size=kan_out_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # ---- Output layer ----
        hidden_dim = rnn_hidden * (2 if bidirectional else 1)
        self.fc_out = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        """
        x: (batch, context_len) یا (batch, context_len, 1)
        output: (batch, pred_len)
        """
        # Handle possible extra channel dimension
        if x.dim() == 2:
            b, L = x.shape
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
            b, L = x.shape
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        assert L == self.context_len, f"Expected context_len={self.context_len}, got {L}"

        # Split context into num_steps segments: (b, T, seq_step)
        x_steps = x.view(b, self.num_steps, self.seq_step)

        # Apply ReLU-KAN to each step:
        # reshape to (b*T, seq_step)
        x_flat = x_steps.reshape(b * self.num_steps, self.seq_step)

        # kan_out: (b*T, kan_out_dim)
        kan_out = self.kan(x_flat)

        # back to sequence form for LSTM: (b, T, kan_out_dim)
        kan_seq = kan_out.view(b, self.num_steps, -1)

        # LSTM
        rnn_out, _ = self.lstm(kan_seq)   # (b, T, hidden*dir)
        last_out = rnn_out[:, -1, :]      # (b, hidden*dir)

        # Projection to prediction length
        return self.fc_out(last_out)      # (b, pred_len)

# Training utilities
def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return torch.sqrt(mse(y_true, y_pred))

def compute_stats(loader, model, device):
    """
    Return (MAE, MSE, RMSE) over a DataLoader
    """
    model.eval()
    tot_mae, tot_mse, tot_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            tot_mae += mae(y_pred, y).item() * y.numel()
            tot_mse += mse(y_pred, y).item() * y.numel()
            tot_samples += y.numel()
    return tot_mae / tot_samples, tot_mse / tot_samples, torch.sqrt(torch.tensor(tot_mse / tot_samples))

# Device, model, optimizer, criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Input range for ReLU-KAN:
# If your data is not normalized, we use the actual min/max of P_train
imin = float(np.min(P_train))
imax = float(np.max(P_train))

model = RecurrentReLUKANTS(
    context_len = Context_Len,
    pred_len = Pred_Len,
    g = G_S,
    k = K_S,
    kan_out_dim = H_Dims,
    rnn_hidden = H_Dims,
    seq_step = 10,
    num_layers = 2,   # 2 (or more) LSTM layers make the network “deep”
    bidirectional = False,
    imin = imin,
    imax = imax
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()          # MAE loss (same as paper’s 0.5*(error)^2)

# Training loop with timing
train_history = []  # (MAE, MSE)
test_history = []   # (MAE, MSE)
train_times = []    # seconds per epoch
infer_times = []    # seconds per epoch (on test_loader)
per_iteration_train_times = []  # seconds per iteration

total_train_time = 0.0
for epoch in range(1, N_Epochs + 1):
    epoch_start = time.perf_counter()

    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        start_time = time.perf_counter()  # Start timer for this iteration

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        per_iteration_train_times.append(time.perf_counter() - start_time)

    epoch_time = time.perf_counter() - epoch_start
    train_times.append(epoch_time)
    total_train_time += epoch_time

    # --------- evaluate on training & test -------------
    train_mae, train_mse, train_rmse = compute_stats(train_loader, model, device)
    test_mae, test_mse, test_rmse = compute_stats(test_loader, model, device)
    train_history.append((train_mae, train_mse))
    test_history.append((test_mae, test_mse))

    # --------- inference timing on test_loader -------------
    infer_start = time.perf_counter()
    _ , _ , _ = compute_stats(test_loader, model, device)
    infer_time = time.perf_counter() - infer_start
    infer_times.append(infer_time)

    print(
        f"Epoch {epoch:04d} | "
        f"Loss {running_loss/len(train_loader.dataset):.6f} | "
        f"Train MAE {train_mae:.6f} | Train MSE {train_mse:.6f} | "
        f"Test MAE {test_mae:.6f} | Test MSE {test_mse:.6f} | "
        f"Train time {epoch_time:.2f}s | Inference time {infer_time:.2f}s"
    )

# Final evaluation & speed‑up
final_train_mae, final_train_mse, _ = compute_stats(train_loader, model, device)
final_test_mae, final_test_mse, _ = compute_stats(test_loader, model, device)

def physics_simulation(x_batch):
    """
    Placeholder for the real physics simulation (e.g. HSPICE).
    Here we just sleep for a tiny amount of time to mimic computational load.
    """
    time.sleep(0.0001 * x_batch.shape[0])  # 0.1 ms per sample

physics_start = time.perf_counter()
for x, _ in test_loader:
    physics_simulation(x.to(device))
physics_time = time.perf_counter() - physics_start

# Speed-up: physics time / (average inference time * number of batches)
avg_infer_time = np.mean(infer_times)
speedup = physics_time / (avg_infer_time * len(test_loader))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

results_df = pd.DataFrame({
    "Metric": [
        "Train MAE", "Test MAE",
        "Train MSE", "Test MSE",
        "Speedup (Physics / ReLU-KAN LSTM)",
        "Total Training Time",
        "Test Time",
        "Per-Iteration Training Time (s)",
        "Total number of trainable parameters:",
        "Structure"
    ],
    "Value": [
        final_train_mae, final_test_mae,
        final_train_mse, final_test_mse,
        speedup,
        total_train_time,
        np.mean(infer_times),
        np.mean(per_iteration_train_times),
        count_parameters(model),
        str(model)
    ]
})

print("\n--- Summary of Results (Markdown) ---")
print(results_df.to_markdown(index=False))

# Result table (Markdown)
# Plot training history (MAE / MSE per epoch)
epochs = range(1, N_Epochs + 1)
train_mae_hist = [h[0] for h in train_history]
train_mse_hist = [h[1] for h in train_history]
test_mae_hist = [h[0] for h in test_history]
test_mse_hist = [h[1] for h in test_history]

plt.figure(figsize=(12, 5))
plt.plot(epochs, train_mae_hist, label="Train MAE", color="blue")
plt.plot(epochs, test_mae_hist, label="Test MAE", color="red", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("MAE per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(epochs, train_mse_hist, label="Train MSE", color="green")
plt.plot(epochs, test_mse_hist, label="Test MSE", color="orange", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Full test‑set prediction vs ground truth
model.eval()
full_predicted_Y_test = np.full_like(Y_test, np.nan, dtype=np.float64)

with torch.no_grad():
    for idx in range(len(test_dataset)):
        x_win, _ = test_dataset[idx]  # only need the input
        x_win = x_win.unsqueeze(0).to(device)  # (1, context_len)
        y_pred_win = model(x_win).cpu().squeeze(0).numpy()  # (pred_len,)

        start_idx = idx + Context_Len
        end_idx = start_idx + Pred_Len
        if start_idx < len(Y_test):
            full_predicted_Y_test[start_idx : min(end_idx, len(Y_test))] = \
                y_pred_win[:min(Pred_Len, len(Y_test) - start_idx)]

start_plot = 145
plt.figure(figsize=(15, 6))
plt.plot(np.arange(start_plot, len(Y_test)), Y_test[start_plot:], label="Actual Y", color="blue")
plt.plot(np.arange(start_plot, len(full_predicted_Y_test)), full_predicted_Y_test[start_plot:],
         label="ReLU-KAN LSTM Prediction", color="red", linestyle="--")
plt.xlabel("Time index")
plt.ylabel("Signal value")
plt.title("Full Test Set: Actual vs. ReLU-KAN LSTM Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

