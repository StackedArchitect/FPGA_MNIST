import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import time

# ===========================================================================
# Full-Precision + BatchNorm  (Baseline)
# 2D CNN for MNIST  (FPGA-targeted architecture, Zynq-7020)
#
# This is the baseline model: standard full-precision weights with
# BatchNorm normalisation. Same architecture as TWN+BN and TTQ+BN
# variants for apples-to-apples comparison.
#
# Architecture:
#   Conv1 -> BN1 -> ReLU -> Pool1      [weights: full precision float32]
#   Conv2 -> BN2 -> ReLU -> Pool2
#   Flatten
#   FC1   -> BN3 -> ReLU
#   FC2   -> logits                    [no BN on output layer]
#
# Differences from TWN+BN and TTQ+BN:
#   - Weights are standard float32 (no ternary quantisation)
#   - No Wp/Wn scaling factors
#   - No threshold-based masking
#   - Uses standard nn.Conv2d and nn.Linear layers
#   - Otherwise identical architecture, training setup, and hyperparameters
# ===========================================================================

# ---- Architecture parameters (identical to TWN+BN and TTQ+BN) ----
INPUT_H, INPUT_W, INPUT_CH       = 28, 28, 1
CONV1_OUT_CH, CONV1_KERNEL       = 4, 3
CONV1_OUT_H = INPUT_H - CONV1_KERNEL + 1            # 26
CONV1_OUT_W = INPUT_W - CONV1_KERNEL + 1            # 26
POOL1_SIZE  = 2
POOL1_OUT_H = CONV1_OUT_H // POOL1_SIZE             # 13
POOL1_OUT_W = CONV1_OUT_W // POOL1_SIZE             # 13
CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL = 4, 8, 3
CONV2_OUT_H = POOL1_OUT_H - CONV2_KERNEL + 1        # 11
CONV2_OUT_W = POOL1_OUT_W - CONV2_KERNEL + 1        # 11
POOL2_SIZE  = 2
POOL2_OUT_H = CONV2_OUT_H // POOL2_SIZE             # 5
POOL2_OUT_W = CONV2_OUT_W // POOL2_SIZE             # 5
FLATTEN_SIZE = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH  # 200
FC1_OUT, FC2_OUT                 = 32, 10

FIXED_POINT_SCALE    = 2**16        # Q16.16
BN_EPS               = 1e-5


class MNIST_CNN2D_BN(nn.Module):
    """
    Full-precision 2D CNN with BatchNorm for MNIST.
    Same architecture as TWN+BN and TTQ+BN — only the weight
    representation differs (float32 instead of ternary).
    """
    def __init__(self):
        super().__init__()
        # Standard full-precision Conv and FC layers
        self.conv1 = nn.Conv2d(INPUT_CH,    CONV1_OUT_CH, CONV1_KERNEL)
        self.conv2 = nn.Conv2d(CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL)
        self.fc1   = nn.Linear(FLATTEN_SIZE, FC1_OUT)
        self.fc2   = nn.Linear(FC1_OUT, FC2_OUT)

        # BatchNorm — same configuration as TTQ+BN and TWN+BN
        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CH, eps=BN_EPS, affine=True)
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CH, eps=BN_EPS, affine=True)
        self.bn3 = nn.BatchNorm1d(FC1_OUT,      eps=BN_EPS, affine=True)

        self.pool  = nn.MaxPool2d(POOL1_SIZE)
        self.pool2 = nn.MaxPool2d(POOL2_SIZE)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # (B, 4, 26, 26)
        x = self.pool(x)                          # (B, 4, 13, 13)
        x = self.relu(self.bn2(self.conv2(x)))   # (B, 8, 11, 11)
        x = self.pool2(x)                         # (B, 8,  5,  5)
        x = x.view(-1, FLATTEN_SIZE)              # (B, 200)
        x = self.relu(self.bn3(self.fc1(x)))     # (B, 32)
        return self.fc2(x)                        # (B, 10) logits

    def count_parameters(self):
        """Count total and per-layer trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n  Full-Precision + BN Parameter Summary:")
        print(f"  {'Layer':<10} {'Shape':<22} {'Params':>10}")
        print("  " + "-"*46)
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"  {name:<10} {str(tuple(param.shape)):<22} {param.numel():>10,}")
        print(f"  {'TOTAL':<10} {'':<22} {total:>10,}")
        return total

    def weight_stats(self):
        """Print weight statistics for all layers."""
        print("\n  Full-Precision Weight Statistics:")
        print(f"  {'Layer':<10} {'Shape':<22} {'Min':>10} {'Max':>10} "
              f"{'Mean':>10} {'Std':>10}")
        print("  " + "-"*72)
        for name, layer in [('conv1', self.conv1), ('conv2', self.conv2),
                             ('fc1',   self.fc1),   ('fc2',   self.fc2)]:
            w = layer.weight.detach()
            print(f"  {name:<10} {str(tuple(w.shape)):<22} {w.min().item():>10.5f} "
                  f"{w.max().item():>10.5f} {w.mean().item():>10.5f} "
                  f"{w.std().item():>10.5f}")


# ===========================================================================
# Training setup (identical to TWN+BN and TTQ+BN)
# ===========================================================================
print("=" * 60)
print("  Full-Precision + BN 2D CNN -- MNIST (Baseline)")
print("=" * 60)
print(f"  Weights      : full precision float32")
print(f"  BatchNorm    : after Conv1, Conv2, FC1")
print(f"  Biases       : full precision")
print(f"  Layers       : conv1, conv2, fc1, fc2")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True,  transform=transform, download=True)
test_dataset  = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Device: {device}")

model     = MNIST_CNN2D_BN().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

model.count_parameters()


# ===========================================================================
# Training loop (same structure as TTQ+BN / TWN+BN)
# ===========================================================================
EPOCHS = 15

print(f"\n[INFO] Training for {EPOCHS} epochs")
print(f"{'Epoch':<8} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 44)

best_acc = 0.0
train_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping (same as TTQ+BN / TWN+BN for fair comparison)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item()
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    scheduler.step()

    # ---- Evaluation ----
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            preds          = outputs.argmax(dim=1)
            test_correct  += (preds == labels).sum().item()
            test_total    += labels.size(0)

    train_acc = 100 * total_correct / total_samples
    test_acc  = 100 * test_correct  / test_total
    avg_loss  = total_loss / len(train_loader)

    print(f"  {epoch+1:<6} {avg_loss:>12.4f} {train_acc:>9.2f}% "
          f"{test_acc:>9.2f}%", end="")
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "cnn2d_bn_mnist_model.pth")
        print("  <- best saved", end="")
    print()

train_time = time.time() - train_start

print(f"\n[INFO] Training completed in {train_time:.1f}s")
print(f"[INFO] Best test accuracy: {best_acc:.2f}%")


# ===========================================================================
# Post-training analysis
# ===========================================================================
# Reload best model
model.load_state_dict(torch.load("cnn2d_bn_mnist_model.pth", map_location=device))
model.eval()

model.weight_stats()

# Logit range check
max_logit = 0.0
with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        if i >= 10:
            break
        logits    = model(images.to(device))
        max_logit = max(max_logit, logits.abs().max().item())

q16_max = 32767.9999
status  = "OK" if max_logit < q16_max else "OVERFLOW"
print(f"\n  Q16.16 range check:")
print(f"    max |logit| observed : {max_logit:.4f}")
print(f"    Q16.16 safe limit    : {q16_max:.0f}")
print(f"    Status               : {status}")


# ===========================================================================
# Test on specific image (same as TTQ+BN / TWN+BN)
# ===========================================================================
test_image, test_label = test_dataset[0]
with torch.no_grad():
    logits   = model(test_image.unsqueeze(0).to(device))
    pred     = logits.argmax(dim=1).item()
    logits_np = logits.cpu().numpy().flatten()
    q16_logits = (logits_np * FIXED_POINT_SCALE).astype(np.int64)

print(f"\n  Single-image test (index 0):")
print(f"    True label      : {test_label}")
print(f"    Predicted digit : {pred}")
print(f"    Logits          : {np.array2string(logits_np, precision=4, suppress_small=True)}")
print(f"    Q16.16          : {q16_logits}")
if pred == test_label:
    print("    >>> CORRECT <<<")
else:
    print(f"    >>> WRONG: expected {test_label}, got {pred} <<<")


# ===========================================================================
# Full test set evaluation — per-digit accuracy
# ===========================================================================
print("\n  Per-digit accuracy on full test set:")
print(f"  {'Digit':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
print("  " + "-"*38)

digit_correct = [0] * 10
digit_total   = [0] * 10

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds   = outputs.argmax(dim=1)
        for i in range(labels.size(0)):
            d = labels[i].item()
            digit_total[d] += 1
            if preds[i].item() == d:
                digit_correct[d] += 1

for d in range(10):
    acc = 100 * digit_correct[d] / digit_total[d] if digit_total[d] > 0 else 0
    print(f"  {d:<8} {digit_correct[d]:>8} {digit_total[d]:>8} {acc:>9.2f}%")

overall = 100 * sum(digit_correct) / sum(digit_total)
print(f"  {'Overall':<8} {sum(digit_correct):>8} {sum(digit_total):>8} {overall:>9.2f}%")


# ===========================================================================
# Model size comparison data (for report)
# ===========================================================================
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
weight_params = sum(p.numel() for n, p in model.named_parameters()
                    if 'weight' in n and ('conv' in n or 'fc' in n))
model_size_bytes = os.path.getsize("cnn2d_bn_mnist_model.pth")

print(f"\n  Model Summary:")
print(f"    Total trainable params  : {total_params:,}")
print(f"    Weight params (conv+fc) : {weight_params:,}")
print(f"    Saved model size        : {model_size_bytes:,} bytes ({model_size_bytes/1024:.1f} KB)")
print(f"    Best test accuracy      : {best_acc:.2f}%")
print(f"    Training time           : {train_time:.1f}s")
print(f"    Weight precision        : float32 (32 bits per weight)")
print(f"    Weight memory (conv+fc) : {weight_params * 32:,} bits "
      f"({weight_params * 4:,} bytes)")

print(f"\n{'='*60}")
print(f"  DONE — saved as cnn2d_bn_mnist_model.pth")
print(f"{'='*60}")
