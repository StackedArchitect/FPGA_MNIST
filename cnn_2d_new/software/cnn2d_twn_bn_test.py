import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import math

# ===========================================================================
# Load trained TWN+BN 2D CNN and export test image for Vivado simulation
# Usage: python cnn2d_twn_bn_test_image.py [INDEX]
#   INDEX = test set index (0-9999), default=1
# ===========================================================================

INPUT_H, INPUT_W, INPUT_CH       = 28, 28, 1
CONV1_OUT_CH, CONV1_KERNEL       = 4, 3
POOL1_SIZE   = 2
POOL1_OUT_H  = (INPUT_H - CONV1_KERNEL + 1) // POOL1_SIZE
POOL1_OUT_W  = (INPUT_W - CONV1_KERNEL + 1) // POOL1_SIZE
CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL = 4, 8, 3
POOL2_SIZE   = 2
POOL2_OUT_H  = (POOL1_OUT_H - CONV2_KERNEL + 1) // POOL2_SIZE
POOL2_OUT_W  = (POOL1_OUT_W - CONV2_KERNEL + 1) // POOL2_SIZE
FLATTEN_SIZE = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH
FC1_OUT, FC2_OUT                 = 32, 10

FIXED_POINT_SCALE    = 2**16
TWN_THRESHOLD_FACTOR = 0.05
BN_EPS               = 1e-5
PAD                  = 20


class TernaryQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, threshold):
        mask_pos = (weight >  threshold)
        mask_neg = (weight < -threshold)
        q_weight = mask_pos.float() - mask_neg.float()
        ctx.save_for_backward(mask_pos, mask_neg)
        return q_weight
    @staticmethod
    def backward(ctx, grad_output):
        mask_pos, mask_neg = ctx.saved_tensors
        return grad_output * (mask_pos | mask_neg).float(), None

class TernaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias   = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, x):
        t = TWN_THRESHOLD_FACTOR * self.weight.detach().abs().max()
        return F.conv2d(x, TernaryQuantizeFn.apply(self.weight, t), self.bias)
    @torch.no_grad()
    def get_ternary_info(self):
        t = TWN_THRESHOLD_FACTOR * self.weight.abs().max()
        mp = (self.weight > t); mn = (self.weight < -t)
        codes = mp.int() - mn.int()
        return codes, (codes == 0).float().mean().item()

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, x):
        t = TWN_THRESHOLD_FACTOR * self.weight.detach().abs().max()
        return F.linear(x, TernaryQuantizeFn.apply(self.weight, t), self.bias)
    @torch.no_grad()
    def get_ternary_info(self):
        t = TWN_THRESHOLD_FACTOR * self.weight.abs().max()
        mp = (self.weight > t); mn = (self.weight < -t)
        codes = mp.int() - mn.int()
        return codes, (codes == 0).float().mean().item()

class MNIST_CNN2D_TWN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TernaryConv2d(INPUT_CH,    CONV1_OUT_CH, CONV1_KERNEL)
        self.conv2 = TernaryConv2d(CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL)
        self.fc1   = TernaryLinear(FLATTEN_SIZE, FC1_OUT)
        self.fc2   = TernaryLinear(FC1_OUT, FC2_OUT)
        self.bn1   = nn.BatchNorm2d(CONV1_OUT_CH, eps=BN_EPS, affine=True)
        self.bn2   = nn.BatchNorm2d(CONV2_OUT_CH, eps=BN_EPS, affine=True)
        self.bn3   = nn.BatchNorm1d(FC1_OUT,      eps=BN_EPS, affine=True)
        self.pool  = nn.MaxPool2d(POOL1_SIZE)
        self.pool2 = nn.MaxPool2d(POOL2_SIZE)
        self.relu  = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, FLATTEN_SIZE)
        x = self.relu(self.bn3(self.fc1(x)))
        return self.fc2(x)


# ===========================================================================
# Main
# ===========================================================================
INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f"\n{'='*60}")
print(f"  TWN+BN 2D CNN -- MNIST test image index: {INDEX}")
print(f"{'='*60}")

device = torch.device("cpu")
model  = MNIST_CNN2D_TWN()
model.load_state_dict(torch.load("cnn2d_twn_bn_mnist_model.pth", map_location=device))
model.eval()
print("[INFO] Loaded cnn2d_twn_bn_mnist_model.pth")

print("\n  Layer Statistics:")
print(f"  {'Layer':<8} {'Sparsity':>10} {'Active':>10} {'+1':>6} {'-1':>6}")
print("  " + "-"*44)
for name, layer in [('conv1', model.conv1), ('conv2', model.conv2),
                    ('fc1',   model.fc1),   ('fc2',   model.fc2)]:
    codes, sp = layer.get_ternary_info()
    n_pos = (codes ==  1).sum().item()
    n_neg = (codes == -1).sum().item()
    print(f"  {name:<8} {sp*100:>9.1f}% {n_pos+n_neg:>5}/{codes.numel()}  "
          f"{n_pos:>5}  {n_neg:>5}")

transform    = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_image, test_label = test_dataset[INDEX]

with torch.no_grad():
    logits     = model(test_image.unsqueeze(0))
    pred       = logits.argmax(dim=1).item()
    logits_np  = logits.cpu().numpy().flatten()
    q16_logits = (logits_np * FIXED_POINT_SCALE).astype(np.int64)

print(f"\nTrue label      : {test_label}")
print(f"Predicted digit : {pred}")
print(f"Logits          : {logits_np}")
print(f"Q16.16 logits   : {q16_logits}")

q16_max   = 32767.9999
max_logit = np.abs(logits_np).max()
if max_logit > q16_max:
    print(f">>> WARNING: Q16.16 overflow! max|logit|={max_logit:.2f} > {q16_max:.0f} <<<")
else:
    print(f">>> Q16.16 range OK: max|logit|={max_logit:.4f} <<<")

if pred == test_label:
    print(">>> Software prediction: CORRECT <<<")
else:
    print(f">>> Software prediction: WRONG (expected {test_label}, got {pred}) <<<")


def to_fixed_point_hex(value):
    fixed = int(round(value * FIXED_POINT_SCALE))
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF
    return format(fixed, '08X')

CNN2D_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "weights")
os.makedirs(CNN2D_WEIGHTS_DIR, exist_ok=True)

image_np   = test_image.squeeze().numpy()
hex_pixels = [to_fixed_point_hex(image_np[r, c])
              for r in range(INPUT_H) for c in range(INPUT_W)]

with open(os.path.join(CNN2D_WEIGHTS_DIR, "data_in.mem"), 'w') as f:
    f.write('\n'.join(hex_pixels))
with open(os.path.join(CNN2D_WEIGHTS_DIR, "expected_label.mem"), 'w') as f:
    f.write(format(test_label, '08X'))

print(f"\nExported to cnn2d_weights/:")
print(f"  data_in.mem        ({len(hex_pixels)} pixels, row-major 28x28)")
print(f"  expected_label.mem (label={test_label})")
print(f"\nRerun Vivado simulation with 'run 200000ns' to verify hardware match.")