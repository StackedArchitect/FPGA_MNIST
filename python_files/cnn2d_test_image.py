import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys

# ===========================================================================
# Load trained 2D CNN model and export test image for Vivado simulation
# Usage: python cnn2d_test_image.py [INDEX]
#   INDEX = test set index (0-9999), default=1
# ===========================================================================

INPUT_H = 28; INPUT_W = 28; INPUT_CH = 1
CONV1_OUT_CH = 4; CONV1_KERNEL = 3
CONV1_OUT_H = 26; CONV1_OUT_W = 26
POOL1_SIZE = 2; POOL1_OUT_H = 13; POOL1_OUT_W = 13
CONV2_IN_CH = 4; CONV2_OUT_CH = 8; CONV2_KERNEL = 3
CONV2_OUT_H = 11; CONV2_OUT_W = 11
POOL2_SIZE = 2; POOL2_OUT_H = 5; POOL2_OUT_W = 5
FLATTEN_SIZE = 200; FC1_OUT = 32; FC2_OUT = 10
PAD = 20; FIXED_POINT_SCALE = 2**16

# ---- Model definition (must match training) ----
class MNIST_CNN2D(nn.Module):
    def __init__(self):
        super(MNIST_CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.pool  = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(200, 32)
        self.fc2   = nn.Linear(32, 10)

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

# ---- Pick test image index ----
INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f"\n{'='*60}")
print(f"  Testing 2D CNN with MNIST test image index: {INDEX}")
print(f"{'='*60}")

# ---- Load model ----
device = torch.device("cpu")
model = MNIST_CNN2D()
model.load_state_dict(torch.load("cnn2d_mnist_model.pth", map_location=device))
model.eval()
print("[INFO] Loaded cnn2d_mnist_model.pth")

# ---- Load test dataset ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

test_image, test_label = test_dataset[INDEX]

# ---- Run inference ----
with torch.no_grad():
    logits = model(test_image.unsqueeze(0))   # (1, 1, 28, 28)
    pred = logits.argmax(dim=1).item()
    logits_np = logits.cpu().numpy().flatten()
    q16_logits = (logits_np * FIXED_POINT_SCALE).astype(np.int64)

print(f"\nTrue label:      {test_label}")
print(f"Predicted digit: {pred}")
print(f"Logits:          {logits_np}")
print(f"Q16.16 logits:   {q16_logits}")

if pred == test_label:
    print(">>> Software prediction: CORRECT <<<")
else:
    print(f">>> Software prediction: WRONG (expected {test_label}, got {pred}) <<<")

# ---- Export data_in.mem and expected_label.mem ----
CNN2D_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "cnn2d_weights")
os.makedirs(CNN2D_WEIGHTS_DIR, exist_ok=True)

def to_fixed_point_hex(value):
    fixed = int(round(value * FIXED_POINT_SCALE))
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF
    return format(fixed, '08X')

image_np = test_image.squeeze().numpy()   # 28×28
hex_pixels = []
for r in range(INPUT_H):
    for c in range(INPUT_W):
        hex_pixels.append(to_fixed_point_hex(image_np[r, c]))

with open(os.path.join(CNN2D_WEIGHTS_DIR, "data_in.mem"), 'w') as f:
    f.write('\n'.join(hex_pixels))

with open(os.path.join(CNN2D_WEIGHTS_DIR, "expected_label.mem"), 'w') as f:
    f.write(format(test_label, '08X'))

print(f"\nExported to cnn2d_weights/:")
print(f"  data_in.mem        ({len(hex_pixels)} pixels, row-major 28×28)")
print(f"  expected_label.mem (label={test_label})")
print(f"\nRerun Vivado simulation with 'run 200000ns' to verify hardware match.")
