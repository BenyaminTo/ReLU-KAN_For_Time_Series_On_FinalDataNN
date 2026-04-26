"""Installation and Imports"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Connecting to Dataset"""
excel_path = "C:/Users/Valieasr/Documents/KAN/FinalDataNN.xlsx"

df = pd.read_excel(excel_path)

# Drop rows with any NaN values in the relevant columns to avoid propagation during training
df_cleaned = df.dropna(subset=["P", "Y", "Ptest", "Ytest"])

print(df_cleaned.head())
print(df_cleaned.columns)

P_train = df_cleaned["P"].values.astype("float32")
Y_train = df_cleaned["Y"].values.astype("float32")

P_test  = df_cleaned["Ptest"].values.astype("float32")
Y_test  = df_cleaned["Ytest"].values.astype("float32")

print("Train len (P,Y):", len(P_train), len(Y_train))
print("Test  len (Ptest,Ytest):", len(P_test), len(Y_test))

"""Displaying Time Series"""
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

"""Setting The Hyperparameters"""
K_S = 5
G_S = 3
Batch_Size = 64
Context_Len = 150
Pred_Len = 50
H_Dims = 32
N_Epochs = 5000

"""Sliding Window dataset with P input and Y output"""
class PairedTimeSeriesDataset(Dataset):
    def __init__(self, x_series, y_series, context_len= Context_Len, pred_len= Pred_Len):
        self.x = torch.tensor(x_series, dtype=torch.float32)
        self.y = torch.tensor(y_series, dtype=torch.float32)
        self.context_len = context_len
        self.pred_len = pred_len
        self.total_len = context_len + pred_len
        self.n_samples = len(self.x) - self.total_len + 1

    def __len__(self):
        return max(self.n_samples, 0)

    def __getitem__(self, idx):
        x_win = self.x[idx : idx + self.context_len]                  # Input from P
        y_win = self.y[idx + self.context_len : idx + self.total_len] # Target from Y
        return x_win, y_win

context_len = Context_Len
pred_len = Pred_Len

train_dataset = PairedTimeSeriesDataset(P_train, Y_train, context_len, pred_len)
test_dataset  = PairedTimeSeriesDataset(P_test,  Y_test,  context_len, pred_len)

print("Train windows:", len(train_dataset))
print("Test  windows:", len(test_dataset))

batch_size = Batch_Size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)

"""ReLU-KAN layer and ReLU-KAN network for time series"""
class ReLUKANLayer(nn.Module):
    def __init__(self, input_size, g, k, output_size, is_train=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.g = g # Interpreted as a hidden dimension for an internal MLP
        self.k = k # Parameter for KAN, can be used for complexity or basis functions

        # A simplified interpretation for a KAN-like layer using ReLU activations.
        # This creates a small MLP for each output, which is then summed or combined.
        # For this basic implementation, we'll use a feed-forward structure:
        self.fc1 = nn.Linear(input_size, g)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(g, output_size)

    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # The RecurrentReLUKANTS expects output shape (batch_size, kan_out_dim, 1)
        return x.unsqueeze(-1)

class RecurrentReLUKANTS(nn.Module):
    """
    Input: (batch, context_len)
    1) Divide into a sequence of length seq_step
    2) Apply ReLUKANLayer to each step
    3) Pass through LSTM
    4) Project to pred_len output
    """
    def __init__(self,
                 context_len: int,
                 pred_len: int,
                 g: int,
                 k: int,
                 kan_out_dim: int,
                 rnn_hidden: int,
                 seq_step: int = 10,   #The length of each time step within the window.
                 num_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()
        assert context_len % seq_step == 0 #context_len should be a multiple of seq_step.
        self.context_len = context_len
        self.pred_len = pred_len
        self.seq_step = seq_step
        self.num_steps = context_len // seq_step

        #Each step is a vector of length seq_step.
        self.kan = ReLUKANLayer(input_size=seq_step,
                                g=g,
                                k=k,
                                output_size=kan_out_dim,
                                is_train=False)  #If you want it to be trainable, set it to True.

        self.rnn_input_dim = kan_out_dim
        self.bidirectional = bidirectional
        self.rnn_hidden = rnn_hidden * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(input_size=self.rnn_input_dim,
                            hidden_size=rnn_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        #Output of the last LSTM → pred_len values.
        self.fc_out = nn.Linear(self.rnn_hidden, pred_len)

    def forward(self, x):
        # x: (batch, context_len)
        if x.dim() == 2:
            b, L = x.shape
        elif x.dim() == 3 and x.size(-1) == 1:
            #If you are using DataLoader with the shape (batch, context_len, 1),
            x = x.squeeze(-1)
            b, L = x.shape
        else:
            raise ValueError("x shape must be (batch, context_len) or (batch, context_len, 1)")

        assert L == self.context_len, f"Expected context_len={self.context_len}, got {L}"

        #"Divide the window into num_steps segments of length seq_step:
        #x: (b, context_len) → (b, num_steps, seq_step)
        x_steps = x.view(b, self.num_steps, self.seq_step)  # (b, T, D_step)

        #"Apply ReLUKAN to each step.
        #First, concatenate everything into the batch dimension.
        x_flat = x_steps.reshape(b * self.num_steps, self.seq_step)      # (b*T, seq_step)
        kan_out = self.kan(x_flat)                                       # (b*T, kan_out_dim, 1)
        kan_out = kan_out.squeeze(-1)                                    # (b*T, kan_out_dim)
        kan_seq = kan_out.view(b, self.num_steps, -1)                    # (b, T, kan_out_dim)

        #Pass through the LSTM.
        rnn_out, (h_n, c_n) = self.lstm(kan_seq)                         # rnn_out: (b, T, hidden*dir)

        # Use the last time step.
        last_out = rnn_out[:, -1, :]                                     # (b, hidden*dir)

        # Projection to pred_len.
        y_pred = self.fc_out(last_out)                                   # (b, pred_len)

        return y_pred

"""Model preparation, optimizer, and evaluation function"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

kan_out_dim = H_Dims
rnn_hidden = H_Dims
seq_step = 10               #Make sure that Context_Len is a multiple of this.

model = RecurrentReLUKANTS(
    context_len=context_len,
    pred_len=pred_len,
    g=G_S,
    k=K_S,
    kan_out_dim=kan_out_dim,
    rnn_hidden=rnn_hidden,
    seq_step=seq_step,
    num_layers=1,
    bidirectional=False
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()  # MAE

def evaluate(loader):
    model.eval()
    mae = 0.0
    mse = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            mae += F.l1_loss(y_pred, y, reduction="sum").item()
            mse += F.mse_loss(y_pred, y, reduction="sum").item()
            n += y.numel()
    return mae / n, mse / n

"""Training loop and MAE/MSE reporting on Train and Test"""
n_epochs = N_Epochs

train_history = []
test_history = []

for epoch in range(1, n_epochs + 1):
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    train_mae, train_mse = evaluate(train_loader)
    test_mae, test_mse = evaluate(test_loader)

    train_history.append((train_mae, train_mse))
    test_history.append((test_mae, test_mse))

    print(
        f"Epoch {epoch:03d} | "
        f"TrainLoss {running_loss/len(train_loader.dataset):.4f} "
        f"| TrainMAE {train_mae:.5f} | TestMAE {test_mae:.5f}"
    )

"""Visualizing Predictions"""
model.eval()

# Initialize an array to store the full reconstructed predictions
# We fill it with NaN initially so unpredicted parts are clearly visible
full_predicted_Y_test = np.full_like(Y_test, np.nan, dtype=np.float64)

# Iterate through the test dataset to get predictions for each window
with torch.no_grad():
    for idx in range(len(test_dataset)):
        x_win, _ = test_dataset[idx]  # We only need x_win for prediction
        x_win = x_win.unsqueeze(0).to(device)  # Add batch dimension for the model
        y_pred_win = model(x_win).cpu().squeeze(0).numpy()  # Get prediction and remove batch dim

        # Place the prediction into the full_predicted_Y_test array
        # This strategy uses the prediction from the latest window for overlapping steps
        start_idx = idx + context_len
        end_idx = start_idx + pred_len

        # Ensure we don't go out of bounds of Y_test
        if start_idx < len(Y_test):
            full_predicted_Y_test[start_idx: min(end_idx, len(Y_test))] = y_pred_win[
                :min(pred_len, len(Y_test) - start_idx)]

start = 145  # Starting point on the time axis
plt.figure(figsize=(15, 6))
# Plot the actual Y_test series
plt.plot(np.arange(start, len(Y_test)),
         Y_test[start:], label="Actual Y_test", color="blue")

# Plot the KAN model's reconstructed predictions
# We start plotting from the point where the first prediction would apply
# The initial context_len steps of Y_test are not directly predicted in this manner
plt.plot(np.arange(start, len(full_predicted_Y_test)),
         full_predicted_Y_test[start:],
         label="KAN Full Prediction", color="red", linestyle='--')

plt.xlabel("Time Step")
plt.ylabel("Normalized Value")
plt.title("Full Test Set: Actual vs. KAN Predicted Y (from t=150)")
plt.legend()
plt.grid(True)
plt.show()

"""Error History Plot"""
# Extract MAE values from history and convert to percentage
train_mae_percent = [mae * 100 for mae, _ in train_history]
test_mae_percent = [mae * 100 for mae, _ in test_history]

epochs = range(1, len(train_mae_percent) + 1)

plt.figure(figsize=(100, 25))
plt.plot(epochs, train_mae_percent, label='Train MAE %', color='blue')
plt.plot(epochs, test_mae_percent, label='Test MAE %', color='red', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (%)')
plt.title('Train and Test MAE Percentage per Epoch')
plt.legend()
plt.grid(True)
plt.xticks(epochs) # Show all epochs on x-axis
plt.show()