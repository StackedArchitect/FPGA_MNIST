import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# ===========================================================================
# Neural Network Architecture: 784 → 256 → 128 → 64 → 10
#
# 4-layer MLP for MNIST digit classification.
# - Layers 1-3: Linear + ReLU
# - Layer 4:    Linear (raw logits for CrossEntropyLoss)
#
# Weights are exported in Q16.16 fixed-point format for FPGA inference.
# Each layer's weights are saved as ONE .mem file (all neurons concatenated,
# each neuron's weights padded with 20 zeros on each side).
# ===========================================================================

# ---- Architecture parameters ----
LAYER_SIZES = [784, 256, 128, 64, 10]   # Input → Output
PAD = 20                                 # Zero-padding on each side of weights
FIXED_POINT_SCALE = 2**16                # Q16.16 format


class MNIST_NN(nn.Module):
    def __init__(self):
        super(MNIST_NN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No activation — raw logits for CrossEntropyLoss
        return x


# ===========================================================================
# Training
# ===========================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # Maps [0,1] → [-1,1]
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset  = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_NN().to(device)
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
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 784).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved as mnist_model.pth")


# ===========================================================================
# Verify a single test image (so user can compare with FPGA output)
# ===========================================================================
test_image, test_label = test_dataset[0]  # Same index used in input.py
with torch.no_grad():
    logits = model(test_image.view(1, -1).to(device))
    pred = logits.argmax(dim=1).item()
    print(f"\n--- Single-image verification (test[0]) ---")
    print(f"True label: {test_label}")
    print(f"Predicted:  {pred}")
    print(f"Logits:     {logits.cpu().numpy().flatten()}")
    print(f"Q16.16 logits: {(logits.cpu().numpy().flatten() * FIXED_POINT_SCALE).astype(np.int64)}")


# ===========================================================================
# Export weights to .mem files for FPGA
# ===========================================================================
def to_fixed_point_hex(value, scale=FIXED_POINT_SCALE):
    """Convert float to Q16.16 fixed-point, return 8-char hex string (32-bit)."""
    fixed = int(round(value * scale))
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF
    return format(fixed, '08X')


def export_layer_weights(layer_name, weight_tensor, num_inputs, out_filename):
    """
    Export one layer's weight matrix to a single .mem file.

    Format (row-major, one neuron after another):
      neuron_0: [20 zeros] [num_inputs weights] [20 zeros]
      neuron_1: [20 zeros] [num_inputs weights] [20 zeros]
      ...
    """
    weights_np = weight_tensor.cpu().numpy()        # Shape: (num_neurons, num_inputs)
    num_neurons = weights_np.shape[0]
    padding = ["00000000"] * PAD

    lines = []
    for n in range(num_neurons):
        hex_weights = [to_fixed_point_hex(w) for w in weights_np[n]]
        lines.extend(padding + hex_weights + padding)

    with open(out_filename, 'w') as f:
        f.write('\n'.join(lines))

    entries_per_neuron = PAD + num_inputs + PAD
    print(f"Generated {out_filename}  ({num_neurons} neurons × {entries_per_neuron} "
          f"= {len(lines)} entries)")


def export_biases(bias_tensor, out_filename):
    """Export biases as Q16.16 32-bit hex values."""
    biases_np = bias_tensor.cpu().numpy()
    lines = [to_fixed_point_hex(b) for b in biases_np]
    with open(out_filename, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Generated {out_filename}  ({len(lines)} biases)")


state_dict = model.state_dict()

print("\n--- Exporting weights and biases ---")

# Layer 1: 784 → 256
export_layer_weights("fc1", state_dict['fc1.weight'], 784, "../w1.mem")
export_biases(state_dict['fc1.bias'], "../b1.mem")

# Layer 2: 256 → 128
export_layer_weights("fc2", state_dict['fc2.weight'], 256, "../w2.mem")
export_biases(state_dict['fc2.bias'], "../b2.mem")

# Layer 3: 128 → 64
export_layer_weights("fc3", state_dict['fc3.weight'], 128, "../w3.mem")
export_biases(state_dict['fc3.bias'], "../b3.mem")

# Layer 4: 64 → 10
export_layer_weights("fc4", state_dict['fc4.weight'], 64, "../w4.mem")
export_biases(state_dict['fc4.bias'], "../b4.mem")


# ===========================================================================
# Also generate data_in.mem and expected_label.mem for the first test image
# ===========================================================================
print("\n--- Generating test input ---")
image_np = test_image.squeeze().numpy()   # 28×28, range [-1, 1]
image_fixed = (image_np * FIXED_POINT_SCALE).astype(np.int32)
hex_pixels = [format(int(val) & 0xFFFFFFFF, '08X') for row in image_fixed for val in row]
padding_data = ["00000000"] * PAD
data_in_lines = padding_data + hex_pixels + padding_data

with open("../data_in.mem", 'w') as f:
    f.write('\n'.join(data_in_lines))
print(f"Generated ../data_in.mem  ({len(data_in_lines)} entries)")

with open("../expected_label.mem", 'w') as f:
    f.write(format(test_label, '08X'))
print(f"Generated ../expected_label.mem  (label={test_label})")

print(f"\nAll .mem files generated successfully!")
print(f"Architecture: {' → '.join(str(s) for s in LAYER_SIZES)}")
print(f"Format: Q16.16 fixed-point, {PAD}-zero padding per side")
