import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os

# ===========================================================================
# 2D CNN Architecture for MNIST  (FPGA-targeted)
#
#   Input:  28×28×1  (single-channel grayscale image)
#   Conv1:  1  → 4  filters, kernel 3×3, valid  → 26×26×4,  ReLU
#   Pool1:  MaxPool2D(2×2)                       → 13×13×4
#   Conv2:  4  → 8  filters, kernel 3×3, valid  → 11×11×8,  ReLU
#   Pool2:  MaxPool2D(2×2)                       → 5×5×8
#   Flatten:                                     → 200
#   FC1:    200 → 32,  ReLU
#   FC2:     32 → 10   (raw logits for CrossEntropyLoss)
#
# Weight export format: Q16.16 fixed-point, 32-bit hex values.
# Conv2D kernels saved per-filter flattened (out_ch, in_ch, kH, kW);
# FC weights use the same padded format as the original MLP.
# ===========================================================================

# ---- Architecture parameters ----
INPUT_H       = 28
INPUT_W       = 28
INPUT_CH      = 1

CONV1_OUT_CH  = 4
CONV1_KERNEL  = 3
CONV1_OUT_H   = INPUT_H - CONV1_KERNEL + 1   # 26
CONV1_OUT_W   = INPUT_W - CONV1_KERNEL + 1   # 26

POOL1_SIZE    = 2
POOL1_OUT_H   = CONV1_OUT_H // POOL1_SIZE    # 13
POOL1_OUT_W   = CONV1_OUT_W // POOL1_SIZE    # 13

CONV2_IN_CH   = CONV1_OUT_CH                 # 4
CONV2_OUT_CH  = 8
CONV2_KERNEL  = 3
CONV2_OUT_H   = POOL1_OUT_H - CONV2_KERNEL + 1  # 11
CONV2_OUT_W   = POOL1_OUT_W - CONV2_KERNEL + 1  # 11

POOL2_SIZE    = 2
POOL2_OUT_H   = CONV2_OUT_H // POOL2_SIZE    # 5
POOL2_OUT_W   = CONV2_OUT_W // POOL2_SIZE    # 5

FLATTEN_SIZE  = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH  # 5×5×8 = 200

FC1_OUT       = 32
FC2_OUT       = 10

PAD           = 20                       # Zero-padding for FC weight .mem files
FIXED_POINT_SCALE = 2**16                # Q16.16

print("=" * 60)
print("  2D CNN Architecture Summary")
print("=" * 60)
print(f"  Input:   {INPUT_H}×{INPUT_W}×{INPUT_CH}")
print(f"  Conv1:   {INPUT_CH}→{CONV1_OUT_CH}, k={CONV1_KERNEL}×{CONV1_KERNEL}  → {CONV1_OUT_H}×{CONV1_OUT_W}×{CONV1_OUT_CH}")
print(f"  Pool1:   MaxPool({POOL1_SIZE}×{POOL1_SIZE})       → {POOL1_OUT_H}×{POOL1_OUT_W}×{CONV1_OUT_CH}")
print(f"  Conv2:   {CONV2_IN_CH}→{CONV2_OUT_CH}, k={CONV2_KERNEL}×{CONV2_KERNEL}  → {CONV2_OUT_H}×{CONV2_OUT_W}×{CONV2_OUT_CH}")
print(f"  Pool2:   MaxPool({POOL2_SIZE}×{POOL2_SIZE})       → {POOL2_OUT_H}×{POOL2_OUT_W}×{CONV2_OUT_CH}")
print(f"  Flatten: {FLATTEN_SIZE}")
print(f"  FC1:     {FLATTEN_SIZE}→{FC1_OUT}, ReLU")
print(f"  FC2:     {FC1_OUT}→{FC2_OUT} (logits)")
print("=" * 60)


# ===========================================================================
# Model Definition
# ===========================================================================
class MNIST_CNN2D(nn.Module):
    def __init__(self):
        super(MNIST_CNN2D, self).__init__()
        # Conv layers (2D)
        self.conv1 = nn.Conv2d(INPUT_CH, CONV1_OUT_CH, kernel_size=CONV1_KERNEL, bias=True)
        self.conv2 = nn.Conv2d(CONV2_IN_CH, CONV2_OUT_CH, kernel_size=CONV2_KERNEL, bias=True)
        self.pool  = nn.MaxPool2d(kernel_size=POOL1_SIZE)
        self.pool2 = nn.MaxPool2d(kernel_size=POOL2_SIZE)
        self.relu  = nn.ReLU()

        # FC layers
        self.fc1 = nn.Linear(FLATTEN_SIZE, FC1_OUT)
        self.fc2 = nn.Linear(FC1_OUT, FC2_OUT)

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
        x = self.relu(self.conv1(x))      # (batch, 4, 26, 26)
        x = self.pool(x)                  # (batch, 4, 13, 13)
        x = self.relu(self.conv2(x))      # (batch, 8, 11, 11)
        x = self.pool2(x)                 # (batch, 8, 5, 5)
        x = x.view(-1, FLATTEN_SIZE)      # (batch, 200)
        x = self.relu(self.fc1(x))        # (batch, 32)
        x = self.fc2(x)                   # (batch, 10)
        return x


# ===========================================================================
# Training
# ===========================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # Maps [0,1] → [-1,1]
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True,  transform=transform, download=True)
test_dataset  = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN2D().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)        # (batch, 1, 28, 28) — keep 2D!
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


# ===========================================================================
# Evaluation
# ===========================================================================
model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)        # keep (batch, 1, 28, 28)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "cnn2d_mnist_model.pth")
print("Model saved as cnn2d_mnist_model.pth")


# ===========================================================================
# Verify a single test image
# ===========================================================================
test_image, test_label = test_dataset[0]
with torch.no_grad():
    logits = model(test_image.unsqueeze(0).to(device))  # (1, 1, 28, 28)
    pred   = logits.argmax(dim=1).item()
    print(f"\n--- Single-image verification (test[0]) ---")
    print(f"True label: {test_label}")
    print(f"Predicted:  {pred}")
    print(f"Logits:     {logits.cpu().numpy().flatten()}")
    print(f"Q16.16 logits: {(logits.cpu().numpy().flatten() * FIXED_POINT_SCALE).astype(np.int64)}")


# ===========================================================================
# Utility: Convert float → Q16.16 hex string (32-bit, two's complement)
# ===========================================================================
def to_fixed_point_hex(value, scale=FIXED_POINT_SCALE):
    fixed = int(round(value * scale))
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF
    return format(fixed, '08X')


# ===========================================================================
# Export Conv2D weights
#
# PyTorch Conv2d weight shape: (out_ch, in_ch, kH, kW)
# Saved as one .mem file per layer, flattened:
#   filter_0: [in_ch × kH × kW values]
#   filter_1: [in_ch × kH × kW values]
#   ...
# ===========================================================================
def export_conv2d_weights(weight_tensor, out_filename):
    w = weight_tensor.cpu().numpy()  # (out_ch, in_ch, kH, kW)
    out_ch, in_ch, kH, kW = w.shape
    lines = []
    for f in range(out_ch):
        for c in range(in_ch):
            for r in range(kH):
                for k in range(kW):
                    lines.append(to_fixed_point_hex(w[f, c, r, k]))
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(lines))
    print(f"Generated {out_filename}  ({out_ch} filters × {in_ch} ch × {kH}×{kW} kernel = {len(lines)} entries)")


def export_biases(bias_tensor, out_filename):
    b = bias_tensor.cpu().numpy()
    lines = [to_fixed_point_hex(v) for v in b]
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(lines))
    print(f"Generated {out_filename}  ({len(lines)} biases)")


def export_fc_weights(weight_tensor, num_inputs, out_filename):
    """FC weights with PAD zero-padding on each side (same format as MLP)."""
    w = weight_tensor.cpu().numpy()  # (num_neurons, num_inputs)
    num_neurons = w.shape[0]
    padding = ["00000000"] * PAD
    lines = []
    for n in range(num_neurons):
        hex_w = [to_fixed_point_hex(v) for v in w[n]]
        lines.extend(padding + hex_w + padding)
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(lines))
    entries_per_neuron = PAD + num_inputs + PAD
    print(f"Generated {out_filename}  ({num_neurons} neurons × {entries_per_neuron} = {len(lines)} entries)")


# ===========================================================================
# Export all weights and biases into ../cnn2d_weights/ folder
# ===========================================================================
CNN2D_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "cnn2d_weights")
os.makedirs(CNN2D_WEIGHTS_DIR, exist_ok=True)

state_dict = model.state_dict()

print(f"\n--- Exporting weights and biases to {CNN2D_WEIGHTS_DIR}/ ---")

# Conv1: (4, 1, 3, 3) → 36 weights
export_conv2d_weights(state_dict['conv1.weight'], os.path.join(CNN2D_WEIGHTS_DIR, "conv1_w.mem"))
export_biases(state_dict['conv1.bias'], os.path.join(CNN2D_WEIGHTS_DIR, "conv1_b.mem"))

# Conv2: (8, 4, 3, 3) → 288 weights
export_conv2d_weights(state_dict['conv2.weight'], os.path.join(CNN2D_WEIGHTS_DIR, "conv2_w.mem"))
export_biases(state_dict['conv2.bias'], os.path.join(CNN2D_WEIGHTS_DIR, "conv2_b.mem"))

# FC1: (32, 200)
export_fc_weights(state_dict['fc1.weight'], FLATTEN_SIZE, os.path.join(CNN2D_WEIGHTS_DIR, "fc1_w.mem"))
export_biases(state_dict['fc1.bias'], os.path.join(CNN2D_WEIGHTS_DIR, "fc1_b.mem"))

# FC2: (10, 32)
export_fc_weights(state_dict['fc2.weight'], FC1_OUT, os.path.join(CNN2D_WEIGHTS_DIR, "fc2_w.mem"))
export_biases(state_dict['fc2.bias'], os.path.join(CNN2D_WEIGHTS_DIR, "fc2_b.mem"))


# ===========================================================================
# Generate data_in.mem and expected_label.mem in cnn2d_weights/ folder
#
# data_in.mem layout: row-major, 28×28 = 784 values
#   pixel[0][0], pixel[0][1], ..., pixel[0][27],
#   pixel[1][0], ..., pixel[27][27]
# ===========================================================================
print("\n--- Generating test input ---")
image_np = test_image.squeeze().numpy()           # 28×28, range [-1, 1]
hex_pixels = []
for r in range(INPUT_H):
    for c in range(INPUT_W):
        hex_pixels.append(to_fixed_point_hex(image_np[r, c]))

with open(os.path.join(CNN2D_WEIGHTS_DIR, "data_in.mem"), 'w') as f:
    f.write('\n'.join(hex_pixels))
print(f"Generated cnn2d_weights/data_in.mem  ({len(hex_pixels)} pixels, row-major 28×28)")

with open(os.path.join(CNN2D_WEIGHTS_DIR, "expected_label.mem"), 'w') as f:
    f.write(format(test_label, '08X'))
print(f"Generated cnn2d_weights/expected_label.mem  (label={test_label})")

print(f"\nAll .mem files generated in cnn2d_weights/ folder!")
print(f"Format: Q16.16 fixed-point")
print(f"Conv2D weights: flat (filter × in_ch × kH × kW)")
print(f"FC weights:     padded ({PAD} zeros each side)")
print(f"Input data:     row-major 28×28")
