import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# ===========================================================================
# 1D CNN Architecture for MNIST  (FPGA-targeted)
#
#   Input:  784 pixels  (28×28 flattened to 1D, 1 channel)
#   Conv1:  1  → 4  filters, kernel_size=5  → 780 × 4,  ReLU
#   Pool1:  MaxPool1D(4)                     → 195 × 4
#   Conv2:  4  → 8  filters, kernel_size=3  → 193 × 8,  ReLU
#   Pool2:  MaxPool1D(4)                     → 48  × 8
#   Flatten:                                 → 384
#   FC1:    384 → 32,  ReLU
#Great, let's do this step by step! Before I start coding, let me propose a CNN architecture and get your input on the design choices.

#Current MLP problem: w1 alone has 784×256 = 200,704 weights — huge for FPGA.

#Proposed 1D CNN Architecture for MNIST:


#   FC2:    32  → 10   (raw logits for CrossEntropyLoss)
#
# Weight export format: Q16.16 fixed-point, 32-bit hex values.
# Conv kernels are saved per-filter flattened; FC weights use the same
# padded format as the original MLP for compatibility with the MAC units.
# ===========================================================================

# ---- Architecture parameters ----
CONV1_IN_CH   = 1
CONV1_OUT_CH  = 4
CONV1_KERNEL  = 5
CONV1_OUT_LEN = 784 - CONV1_KERNEL + 1   # 780

POOL1_SIZE    = 4
POOL1_OUT_LEN = CONV1_OUT_LEN // POOL1_SIZE  # 195

CONV2_IN_CH   = CONV1_OUT_CH              # 4
CONV2_OUT_CH  = 8
CONV2_KERNEL  = 3
CONV2_OUT_LEN = POOL1_OUT_LEN - CONV2_KERNEL + 1  # 193

POOL2_SIZE    = 4
POOL2_OUT_LEN = CONV2_OUT_LEN // POOL2_SIZE  # 48

FLATTEN_SIZE  = POOL2_OUT_LEN * CONV2_OUT_CH  # 48 × 8 = 384

FC1_OUT       = 32
FC2_OUT       = 10

PAD           = 20                       # Zero-padding for FC weight .mem files
FIXED_POINT_SCALE = 2**16                # Q16.16

print("=" * 60)
print("  CNN Architecture Summary")
print("=" * 60)
print(f"  Input:   784 × 1")
print(f"  Conv1:   {CONV1_IN_CH}→{CONV1_OUT_CH}, k={CONV1_KERNEL}  → {CONV1_OUT_LEN} × {CONV1_OUT_CH}")
print(f"  Pool1:   MaxPool({POOL1_SIZE})       → {POOL1_OUT_LEN} × {CONV1_OUT_CH}")
print(f"  Conv2:   {CONV2_IN_CH}→{CONV2_OUT_CH}, k={CONV2_KERNEL}  → {CONV2_OUT_LEN} × {CONV2_OUT_CH}")
print(f"  Pool2:   MaxPool({POOL2_SIZE})       → {POOL2_OUT_LEN} × {CONV2_OUT_CH}")
print(f"  Flatten: {FLATTEN_SIZE}")
print(f"  FC1:     {FLATTEN_SIZE}→{FC1_OUT}, ReLU")
print(f"  FC2:     {FC1_OUT}→{FC2_OUT} (logits)")
print("=" * 60)


# ===========================================================================
# Model Definition
# ===========================================================================
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv1d(CONV1_IN_CH, CONV1_OUT_CH, kernel_size=CONV1_KERNEL, bias=True)
        self.conv2 = nn.Conv1d(CONV2_IN_CH, CONV2_OUT_CH, kernel_size=CONV2_KERNEL, bias=True)
        self.pool  = nn.MaxPool1d(kernel_size=POOL1_SIZE)   # reused for both pool layers
        self.pool2 = nn.MaxPool1d(kernel_size=POOL2_SIZE)
        self.relu  = nn.ReLU()

        # FC layers
        self.fc1 = nn.Linear(FLATTEN_SIZE, FC1_OUT)
        self.fc2 = nn.Linear(FC1_OUT, FC2_OUT)

    def forward(self, x):
        # x shape: (batch, 784)
        x = x.view(-1, 1, 784)           # (batch, 1, 784) — 1 channel
        x = self.relu(self.conv1(x))      # (batch, 4, 780)
        x = self.pool(x)                  # (batch, 4, 195)
        x = self.relu(self.conv2(x))      # (batch, 8, 193)
        x = self.pool2(x)                 # (batch, 8, 48)
        x = x.view(-1, FLATTEN_SIZE)      # (batch, 384)
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
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.view(-1, 784).to(device)
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
        images = images.view(-1, 784).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "cnn_mnist_model.pth")
print("Model saved as cnn_mnist_model.pth")


# ===========================================================================
# Verify a single test image
# ===========================================================================
test_image, test_label = test_dataset[0]
with torch.no_grad():
    logits = model(test_image.view(1, -1).to(device))
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
# Export Conv1D weights
#
# PyTorch Conv1d weight shape: (out_ch, in_ch, kernel_size)
# Saved as one .mem file per layer, flattened:
#   filter_0: [in_ch × kernel values]
#   filter_1: [in_ch × kernel values]
#   ...
# ===========================================================================
def export_conv_weights(weight_tensor, out_filename):
    w = weight_tensor.cpu().numpy()  # (out_ch, in_ch, kernel_size)
    out_ch, in_ch, ks = w.shape
    lines = []
    for f in range(out_ch):
        for c in range(in_ch):
            for k in range(ks):
                lines.append(to_fixed_point_hex(w[f, c, k]))
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(lines))
    print(f"Generated {out_filename}  ({out_ch} filters × {in_ch} ch × {ks} kernel = {len(lines)} entries)")


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
# Export all weights and biases into ../cnn_weights/ folder
# ===========================================================================
import os

CNN_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "cnn_weights")
os.makedirs(CNN_WEIGHTS_DIR, exist_ok=True)

state_dict = model.state_dict()

print(f"\n--- Exporting weights and biases to {CNN_WEIGHTS_DIR}/ ---")

# Conv1: (4, 1, 5)
export_conv_weights(state_dict['conv1.weight'], os.path.join(CNN_WEIGHTS_DIR, "conv1_w.mem"))
export_biases(state_dict['conv1.bias'], os.path.join(CNN_WEIGHTS_DIR, "conv1_b.mem"))

# Conv2: (8, 4, 3)
export_conv_weights(state_dict['conv2.weight'], os.path.join(CNN_WEIGHTS_DIR, "conv2_w.mem"))
export_biases(state_dict['conv2.bias'], os.path.join(CNN_WEIGHTS_DIR, "conv2_b.mem"))

# FC1: (32, 384)
export_fc_weights(state_dict['fc1.weight'], FLATTEN_SIZE, os.path.join(CNN_WEIGHTS_DIR, "fc1_w.mem"))
export_biases(state_dict['fc1.bias'], os.path.join(CNN_WEIGHTS_DIR, "fc1_b.mem"))

# FC2: (10, 32)
export_fc_weights(state_dict['fc2.weight'], FC1_OUT, os.path.join(CNN_WEIGHTS_DIR, "fc2_w.mem"))
export_biases(state_dict['fc2.bias'], os.path.join(CNN_WEIGHTS_DIR, "fc2_b.mem"))


# ===========================================================================
# Generate data_in.mem and expected_label.mem in cnn_weights/ folder
# ===========================================================================
print("\n--- Generating test input ---")
image_np = test_image.squeeze().numpy()           # 28×28, range [-1, 1]
image_fixed = (image_np * FIXED_POINT_SCALE).astype(np.int32)
hex_pixels = [format(int(val) & 0xFFFFFFFF, '08X') for row in image_fixed for val in row]

# No padding for CNN input — conv kernels slide over raw 784 pixels
with open(os.path.join(CNN_WEIGHTS_DIR, "data_in.mem"), 'w') as f:
    f.write('\n'.join(hex_pixels))
print(f"Generated cnn_weights/data_in.mem  ({len(hex_pixels)} entries, no padding)")

with open(os.path.join(CNN_WEIGHTS_DIR, "expected_label.mem"), 'w') as f:
    f.write(format(test_label, '08X'))
print(f"Generated cnn_weights/expected_label.mem  (label={test_label})")

print(f"\nAll .mem files generated in cnn_weights/ folder!")
print(f"Format: Q16.16 fixed-point")
print(f"Conv weights: flat (filter × channel × kernel)")
print(f"FC weights:   padded ({PAD} zeros each side)")
