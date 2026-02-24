import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys

# ===========================================================================
# Load trained CNN model and export weights for a DIFFERENT test image
# Usage: python cnn_test_image.py [INDEX]
#   INDEX = test set index (0-9999), default=1
# ===========================================================================

CONV1_IN_CH = 1; CONV1_OUT_CH = 4; CONV1_KERNEL = 5
CONV1_OUT_LEN = 780
POOL1_SIZE = 4; POOL1_OUT_LEN = 195
CONV2_IN_CH = 4; CONV2_OUT_CH = 8; CONV2_KERNEL = 3
CONV2_OUT_LEN = 193
POOL2_SIZE = 4; POOL2_OUT_LEN = 48
FLATTEN_SIZE = 384; FC1_OUT = 32; FC2_OUT = 10
PAD = 20; FIXED_POINT_SCALE = 2**16

# ---- Model definition (must match training) ----
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=5)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=3)
        self.pool  = nn.MaxPool1d(kernel_size=4)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(384, 32)
        self.fc2   = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 1, 784)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 384)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---- Pick test image index ----
INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f"\n{'='*60}")
print(f"  Testing with MNIST test image index: {INDEX}")
print(f"{'='*60}")

# ---- Load model ----
device = torch.device("cpu")
model = MNIST_CNN()
model.load_state_dict(torch.load("cnn_mnist_model.pth", map_location=device))
model.eval()
print("[INFO] Loaded cnn_mnist_model.pth")

# ---- Load test dataset ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

test_image, test_label = test_dataset[INDEX]

# ---- Run inference ----
with torch.no_grad():
    logits = model(test_image.view(1, -1))
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
CNN_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "cnn_weights")
os.makedirs(CNN_WEIGHTS_DIR, exist_ok=True)

def to_fixed_point_hex(value):
    fixed = int(round(value * FIXED_POINT_SCALE))
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF
    return format(fixed, '08X')

image_np = test_image.squeeze().numpy()
image_fixed = (image_np * FIXED_POINT_SCALE).astype(np.int32)
hex_pixels = [format(int(val) & 0xFFFFFFFF, '08X') for row in image_fixed for val in row]

with open(os.path.join(CNN_WEIGHTS_DIR, "data_in.mem"), 'w') as f:
    f.write('\n'.join(hex_pixels))

with open(os.path.join(CNN_WEIGHTS_DIR, "expected_label.mem"), 'w') as f:
    f.write(format(test_label, '08X'))

print(f"\nExported to cnn_weights/:")
print(f"  data_in.mem        ({len(hex_pixels)} pixels)")
print(f"  expected_label.mem (label={test_label})")
print(f"\nRerun Vivado simulation with 'run 200000ns' to verify hardware match.")
