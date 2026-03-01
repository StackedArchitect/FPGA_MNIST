"""
mlp_simple_model.py
====================
Train a simple 784 → 10 → 10 MLP on MNIST and export weights to
mlp_weights/ in the format expected by neural_network.sv / tb_neuralnetwork.sv.

Weight file layout (matches tb_neuralnetwork.sv $readmemh calls):
  w1_1.mem  …  w1_10.mem  : Layer-1 weights  (824 x 32-bit Q16.16 each, with ±20-zero padding)
  w2_1.mem  …  w2_10.mem  : Layer-2 weights   (50 x 32-bit Q16.16 each, with ±20-zero padding)
  b1.mem                  : Layer-1 biases    (10 x 32-bit Q16.16)
  b2.mem                  : Layer-2 biases    (10 x 64-bit Q16.16, zero-sign-extended)
  data_in.mem             : Test image        (824 x 32-bit Q16.16, with ±20-zero padding)
  expected_label.mem      : Ground-truth label (1 x 32-bit)
"""

import os, struct, numpy as np, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PAD            = 20            # zeros padding each side of weight row
SCALE          = 2 ** 16       # Q16.16 fixed-point
OUT_DIR        = os.path.join(os.path.dirname(__file__), '..', 'mlp_weights')
TEST_IMAGE_IDX = 100           # which MNIST test image to export

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Architecture  784 → 10 → 10
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10,  10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)           # raw logits
        return x

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_data  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64,  shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=1000, shuffle=False)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
model     = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS    = 10

print("Training 784→10→10 SimpleMLP …")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.view(-1, 784)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"  Epoch {epoch+1:2d}/{EPOCHS}  loss={running_loss/len(train_loader):.4f}")

model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.view(-1, 784))
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)
accuracy = 100 * correct / total
print(f"\nTest accuracy: {accuracy:.2f}%")

# Save checkpoint
torch.save(model.state_dict(), os.path.join(OUT_DIR, 'mlp_simple_model.pth'))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_q1616_32(v):
    """float → signed 32-bit Q16.16 two's-complement int"""
    return int(round(v * SCALE)) & 0xFFFFFFFF

def to_q1616_64(v):
    """float → signed 64-bit Q16.16 two's-complement int (sign-extended)"""
    iv = int(round(v * SCALE))
    return iv & 0xFFFFFFFFFFFFFFFF

def write_mem_32(path, values):
    with open(path, 'w') as f:
        f.write('\n'.join(f'{to_q1616_32(v):08X}' for v in values) + '\n')

def write_mem_64(path, values):
    with open(path, 'w') as f:
        f.write('\n'.join(f'{to_q1616_64(v):016X}' for v in values) + '\n')

def padded_row(weights_1d):
    """Return padded list: [0]*PAD + weights + [0]*PAD"""
    return [0.0] * PAD + list(weights_1d) + [0.0] * PAD

# ---------------------------------------------------------------------------
# Layer 1 weights & biases  (fc1: 784→10)
# ---------------------------------------------------------------------------
W1 = model.fc1.weight.detach().numpy()   # shape (10, 784)
B1 = model.fc1.bias.detach().numpy()     # shape (10,)

LAYER1_NEURON_WIDTH = 823  # 20+784+20-1 → indices 0..823 = 824 entries
assert len(padded_row(W1[0])) == LAYER1_NEURON_WIDTH + 1, \
    f"Expected {LAYER1_NEURON_WIDTH+1} but got {len(padded_row(W1[0]))}"

for i in range(10):
    path = os.path.join(OUT_DIR, f'w1_{i+1}.mem')
    write_mem_32(path, padded_row(W1[i]))
    print(f"  Written {path}  ({LAYER1_NEURON_WIDTH+1} entries)")

write_mem_32(os.path.join(OUT_DIR, 'b1.mem'), B1)
print(f"  Written b1.mem  (10 x 32-bit)")

# ---------------------------------------------------------------------------
# Layer 2 weights & biases  (fc2: 10→10)
# ---------------------------------------------------------------------------
W2 = model.fc2.weight.detach().numpy()   # shape (10, 10)
B2 = model.fc2.bias.detach().numpy()     # shape (10,)

LAYER2_NEURON_WIDTH = 49   # 20+10+20-1 → indices 0..49 = 50 entries
assert len(padded_row(W2[0])) == LAYER2_NEURON_WIDTH + 1, \
    f"Expected {LAYER2_NEURON_WIDTH+1} but got {len(padded_row(W2[0]))}"

for i in range(10):
    path = os.path.join(OUT_DIR, f'w2_{i+1}.mem')
    write_mem_32(path, padded_row(W2[i]))
    print(f"  Written {path}  ({LAYER2_NEURON_WIDTH+1} entries)")

# b2 is 64-bit in neural_network.sv port (sign-extended Q16.16)
write_mem_64(os.path.join(OUT_DIR, 'b2.mem'), B2)
print(f"  Written b2.mem  (10 x 64-bit sign-extended)")

# ---------------------------------------------------------------------------
# Test image (data_in.mem)  — 784-pixel image padded to 824 entries
# ---------------------------------------------------------------------------
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                           transform=transform)
image, label = test_dataset[TEST_IMAGE_IDX]
pixels = image.squeeze().numpy().flatten()   # 784 floats in [-1, 1]

# Build padded data_in (same as input.py but in one script)
padded_pixels = [0.0]*PAD + list(pixels) + [0.0]*PAD     # 824 entries
assert len(padded_pixels) == LAYER1_NEURON_WIDTH + 1

with open(os.path.join(OUT_DIR, 'data_in.mem'), 'w') as f:
    f.write('\n'.join(f'{to_q1616_32(v):08X}' for v in padded_pixels) + '\n')
print(f"\n  Written data_in.mem  (image idx={TEST_IMAGE_IDX}, label={label})")

with open(os.path.join(OUT_DIR, 'expected_label.mem'), 'w') as f:
    f.write(f'{label:08X}\n')
print(f"  Written expected_label.mem  (label={label})")

# ---------------------------------------------------------------------------
# Verify with software inference
# ---------------------------------------------------------------------------
with torch.no_grad():
    t = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0)
    logits = model(t).squeeze().numpy()
pred = int(np.argmax(logits))
print(f"\nSoftware prediction for test image {TEST_IMAGE_IDX}: {pred}  (true label: {label})")
print(f"Logits: {np.round(logits, 3)}")
print(f"\nAll MLP weight files written to: {os.path.abspath(OUT_DIR)}")
