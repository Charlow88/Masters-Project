# ultra_minimal_mlp.py
import os, glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = ["coherent", "squeezed", "cat"]
LAB2ID = {k:i for i,k in enumerate(LABELS)}

def infer_label(name: str) -> str:
    n = name.lower()
    if n.startswith("cat"): return "cat"
    if n.startswith("coh") or n.startswith("coherent"): return "coherent"
    if n.startswith("sq")  or n.startswith("squeezed"): return "squeezed"
    return n.split("_")[0]

# find paths to images that match a label
paths = [p for p in sorted(glob.glob(os.path.join("data/wigner", "*.png")))
         if infer_label(os.path.basename(p)) in LAB2ID]
assert paths, f"No labeled PNGs in {'data/wigner'}"

# convert each image into a vector of pixel darknesses and a label ID
X_list, y_list = [], []
for p in paths:
    y = LAB2ID[infer_label(os.path.basename(p))]
    im = Image.open(p).convert("L").resize((64, 64))
    arr = np.array(im, dtype=np.float32) / 255.0
    m, s = arr.mean(), arr.std()
    arr = (arr - m) / (s + 1e-8)
    X_list.append(arr.reshape(-1))
    y_list.append(y)

# Convert to torch for training
X = torch.from_numpy(np.stack(X_list))           # [N, 4096]
y = torch.tensor(y_list, dtype=torch.long)       # [N]

# Train/val split (80/20) for training and validation
N = X.size(0)
perm = torch.randperm(N)
n_val = int(0.2 * N)
val_idx, train_idx = perm[:n_val], perm[n_val:]
Xtr, ytr = X[train_idx], y[train_idx]
Xva, yva = X[val_idx],  y[val_idx]

# wrap tensors in DataLoader - batching of 64 per step to use mini batch stochastic gradient descent
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=64)

# simple MLP, input layer 4096 neurons, two hidden layers (256, 128), output layer 3 neurons
model = nn.Sequential(
    nn.Linear(64*64, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, len(LABELS))
).to(DEVICE)

# Adam is a gradient optimiser to update weights, CrossEntropyLoss is a common loss function for classification
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

train_loss_hist, train_acc_hist, val_acc_hist = [], [], []

# loop in training the model
EPOCHS = 20
for epoch in range(1, EPOCHS+1):
    model.train()
    tot, correct, loss_sum = 0, 0, 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb) # forward pass to get predictions
        loss = loss_fn(logits, yb) # compute loss
        loss.backward() # backpropagate to get gradients
        opt.step() # update weights
        loss_sum += loss.item() * yb.size(0) 
        correct  += (logits.argmax(1) == yb).sum().item()
        tot      += yb.size(0)
    tr_loss = loss_sum / tot 
    tr_acc  = correct / tot 

# turn off gradient tracking for faster evaluation (switch from training to evaluation mode so no updates to weights)
    model.eval()
    with torch.no_grad():
        vtot, vcorrect = 0, 0
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1) # finds highest activation output neuron as predicted class
            vcorrect += (pred == yb).sum().item() # compares with true label
            vtot     += yb.size(0)
    va_acc = vcorrect / vtot if vtot else 0.0

        # record stats for total loss and accuracy
    train_loss_hist.append(tr_loss)
    train_acc_hist.append(tr_acc)
    val_acc_hist.append(va_acc)

    print(f"epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val acc {va_acc:.3f}")

epochs = range(1, EPOCHS+1)

plt.figure()
plt.plot(epochs, train_acc_hist, label="Train acc")
plt.plot(epochs, val_acc_hist, label="Val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("plots/acc_curve.png")
plt.show()

plt.figure()
plt.plot(epochs, train_loss_hist, label="Train loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("plots/loss_curve.png")
plt.show()