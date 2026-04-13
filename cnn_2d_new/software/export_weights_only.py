"""
Export-only script: loads existing .pth and re-exports all .mem files.
No training — just weight export from the saved model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import math

# ---- Architecture parameters ----
INPUT_H, INPUT_W, INPUT_CH = 28, 28, 1
CONV1_OUT_CH, CONV1_KERNEL = 4, 3
POOL1_SIZE = 2
POOL1_OUT_H = (INPUT_H - CONV1_KERNEL + 1) // POOL1_SIZE
POOL1_OUT_W = (INPUT_W - CONV1_KERNEL + 1) // POOL1_SIZE
CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL = 4, 8, 3
POOL2_SIZE = 2
POOL2_OUT_H = (POOL1_OUT_H - CONV2_KERNEL + 1) // POOL2_SIZE
POOL2_OUT_W = (POOL1_OUT_W - CONV2_KERNEL + 1) // POOL2_SIZE
FLATTEN_SIZE = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH
FC1_OUT, FC2_OUT = 32, 10
FIXED_POINT_SCALE = 2**16
TTQ_THRESHOLD_FACTOR = 0.05
BN_EPS = 1e-5
PAD = 20

# ---- Model definition (must match training script) ----
class TTQQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, wp, wn, threshold):
        mask_pos = (weight >  threshold)
        mask_neg = (weight < -threshold)
        q_weight = mask_pos.float() * wp - mask_neg.float() * wn
        ctx.save_for_backward(mask_pos, mask_neg, wp, wn)
        return q_weight
    @staticmethod
    def backward(ctx, grad_output):
        mask_pos, mask_neg, wp, wn = ctx.saved_tensors
        mask_zero = ~(mask_pos | mask_neg)
        grad_weight = (mask_pos.float() * wp + mask_zero.float() * 1.0
                     + mask_neg.float() * wn) * grad_output
        grad_wp = (grad_output * mask_pos.float()).sum().view(1)
        grad_wn = -(grad_output * mask_neg.float()).sum().view(1)
        return grad_weight, grad_wp, grad_wn, None

class TTQConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.wp = nn.Parameter(torch.ones(1) * 0.1)
        self.wn = nn.Parameter(torch.ones(1) * 0.1)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, x):
        t = TTQ_THRESHOLD_FACTOR * self.weight.detach().abs().max()
        return F.conv2d(x, TTQQuantizeFn.apply(self.weight, self.wp.abs(), self.wn.abs(), t), self.bias)
    @torch.no_grad()
    def get_ternary_info(self):
        t = TTQ_THRESHOLD_FACTOR * self.weight.abs().max()
        wp = self.wp.abs().item(); wn = self.wn.abs().item()
        mp = (self.weight > t); mn = (self.weight < -t)
        codes = mp.int() - mn.int()
        q_weights = mp.float() * wp - mn.float() * wn
        sparsity = (codes == 0).float().mean().item()
        return codes, wp, wn, q_weights, sparsity

class TTQLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.wp = nn.Parameter(torch.ones(1) * 0.1)
        self.wn = nn.Parameter(torch.ones(1) * 0.1)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, x):
        t = TTQ_THRESHOLD_FACTOR * self.weight.detach().abs().max()
        return F.linear(x, TTQQuantizeFn.apply(self.weight, self.wp.abs(), self.wn.abs(), t), self.bias)
    @torch.no_grad()
    def get_ternary_info(self):
        t = TTQ_THRESHOLD_FACTOR * self.weight.abs().max()
        wp = self.wp.abs().item(); wn = self.wn.abs().item()
        mp = (self.weight > t); mn = (self.weight < -t)
        codes = mp.int() - mn.int()
        q_weights = mp.float() * wp - mn.float() * wn
        sparsity = (codes == 0).float().mean().item()
        return codes, wp, wn, q_weights, sparsity

class MNIST_CNN2D_TTQ_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TTQConv2d(INPUT_CH, CONV1_OUT_CH, CONV1_KERNEL)
        self.conv2 = TTQConv2d(CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL)
        self.fc1 = TTQLinear(FLATTEN_SIZE, FC1_OUT)
        self.fc2 = TTQLinear(FC1_OUT, FC2_OUT)
        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CH, eps=BN_EPS, affine=True)
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CH, eps=BN_EPS, affine=True)
        self.bn3 = nn.BatchNorm1d(FC1_OUT, eps=BN_EPS, affine=True)
        self.pool = nn.MaxPool2d(POOL1_SIZE)
        self.pool2 = nn.MaxPool2d(POOL2_SIZE)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, FLATTEN_SIZE)
        x = self.relu(self.bn3(self.fc1(x)))
        return self.fc2(x)

# ---- Utility functions ----
def to_fixed_point_hex(value, scale=FIXED_POINT_SCALE):
    fixed = int(round(value * scale))
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF
    return format(fixed, '08X')

def to_ternary_2bit(code):
    return {0: "0", 1: "1", -1: "3"}[int(code)]

def get_folded_bn_params(bn_layer):
    gamma = bn_layer.weight.detach().cpu().numpy()
    beta  = bn_layer.bias.detach().cpu().numpy()
    mean  = bn_layer.running_mean.detach().cpu().numpy()
    var   = bn_layer.running_var.detach().cpu().numpy()
    eps   = bn_layer.eps
    scale = gamma / np.sqrt(var + eps)
    shift = beta - mean * scale
    return scale, shift

CNN2D_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights_ttq")
os.makedirs(CNN2D_WEIGHTS_DIR, exist_ok=True)
def fp(name): return os.path.join(CNN2D_WEIGHTS_DIR, name)

def export_ttq_conv_with_bn(conv_layer, bn_layer, layer_name):
    codes, wp_val, wn_val, _, sparsity = conv_layer.get_ternary_info()
    w = codes.cpu().numpy()
    out_ch, in_ch, kH, kW = w.shape
    code_lines = []
    for f in range(out_ch):
        for c in range(in_ch):
            for r in range(kH):
                for k in range(kW):
                    code_lines.append(to_ternary_2bit(int(w[f, c, r, k])))
    with open(fp(f"{layer_name}_ternary_codes.mem"), 'w') as f_:
        f_.write('\n'.join(code_lines))
    with open(fp(f"{layer_name}_wp.mem"), 'w') as f_:
        f_.write(to_fixed_point_hex(wp_val))
    with open(fp(f"{layer_name}_wn.mem"), 'w') as f_:
        f_.write(to_fixed_point_hex(wn_val))
    b = conv_layer.bias.detach().cpu().numpy()
    with open(fp(f"{layer_name}_b.mem"), 'w') as f_:
        f_.write('\n'.join(to_fixed_point_hex(v) for v in b))
    scale, shift = get_folded_bn_params(bn_layer)
    with open(fp(f"{layer_name}_bn_scale.mem"), 'w') as f_:
        f_.write('\n'.join(to_fixed_point_hex(v) for v in scale))
    with open(fp(f"{layer_name}_bn_shift.mem"), 'w') as f_:
        f_.write('\n'.join(to_fixed_point_hex(v) for v in shift))
    n_pos = (codes == 1).sum().item()
    n_neg = (codes == -1).sum().item()
    print(f"  {layer_name:<8} | codes: {len(code_lines)}, Wp={wp_val:.5f}, Wn={wn_val:.5f}, "
          f"sparsity={sparsity*100:.1f}% (+1:{n_pos}, -1:{n_neg})")

def export_ttq_fc_with_bn(fc_layer, bn_layer, layer_name, num_inputs):
    codes, wp_val, wn_val, _, sparsity = fc_layer.get_ternary_info()
    w = codes.cpu().numpy()
    num_neurons = w.shape[0]
    padding = ["0"] * PAD
    code_lines = []
    for n in range(num_neurons):
        row = [to_ternary_2bit(int(v)) for v in w[n]]
        code_lines.extend(padding + row + padding)
    with open(fp(f"{layer_name}_ternary_codes.mem"), 'w') as f_:
        f_.write('\n'.join(code_lines))
    with open(fp(f"{layer_name}_wp.mem"), 'w') as f_:
        f_.write(to_fixed_point_hex(wp_val))
    with open(fp(f"{layer_name}_wn.mem"), 'w') as f_:
        f_.write(to_fixed_point_hex(wn_val))
    b = fc_layer.bias.detach().cpu().numpy()
    with open(fp(f"{layer_name}_b.mem"), 'w') as f_:
        f_.write('\n'.join(to_fixed_point_hex(v) for v in b))
    if bn_layer is not None:
        scale, shift = get_folded_bn_params(bn_layer)
        with open(fp(f"{layer_name}_bn_scale.mem"), 'w') as f_:
            f_.write('\n'.join(to_fixed_point_hex(v) for v in scale))
        with open(fp(f"{layer_name}_bn_shift.mem"), 'w') as f_:
            f_.write('\n'.join(to_fixed_point_hex(v) for v in shift))
    n_pos = (codes == 1).sum().item()
    n_neg = (codes == -1).sum().item()
    print(f"  {layer_name:<8} | codes: {len(code_lines)}, Wp={wp_val:.5f}, Wn={wn_val:.5f}, "
          f"sparsity={sparsity*100:.1f}% (+1:{n_pos}, -1:{n_neg})")

# ============================================================
# MAIN
# ============================================================
INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

print(f"\n{'='*60}")
print(f"  TTQ+BN 2D CNN -- EXPORT ONLY (no training)")
print(f"  Test image index: {INDEX}")
print(f"{'='*60}")

device = torch.device("cpu")
model = MNIST_CNN2D_TTQ_BN()
model.load_state_dict(torch.load("cnn2d_ttq_bn_mnist_model.pth", map_location=device))
model.eval()
print("[INFO] Loaded cnn2d_ttq_bn_mnist_model.pth\n")

# Export all weights
print("--- Exporting all .mem files ---")
export_ttq_conv_with_bn(model.conv1, model.bn1, "conv1")
export_ttq_conv_with_bn(model.conv2, model.bn2, "conv2")
export_ttq_fc_with_bn(model.fc1, model.bn3, "fc1", FLATTEN_SIZE)
export_ttq_fc_with_bn(model.fc2, None, "fc2", FC1_OUT)

# Export test image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True)
test_image, test_label = test_dataset[INDEX]

image_np = test_image.squeeze().numpy()
hex_pixels = [to_fixed_point_hex(image_np[r, c])
              for r in range(INPUT_H) for c in range(INPUT_W)]
with open(fp("data_in.mem"), 'w') as f_:
    f_.write('\n'.join(hex_pixels))
with open(fp("expected_label.mem"), 'w') as f_:
    f_.write(format(test_label, '08X'))

# Software inference
with torch.no_grad():
    logits = model(test_image.unsqueeze(0))
    pred = logits.argmax(dim=1).item()
    logits_np = logits.cpu().numpy().flatten()
    q16_logits = (logits_np * FIXED_POINT_SCALE).astype(np.int64)

print(f"\n--- Software reference ---")
print(f"True label      : {test_label}")
print(f"Predicted digit : {pred}")
print(f"Q16.16 logits   : {q16_logits}")
print(f"\n[DONE] All .mem files exported to {CNN2D_WEIGHTS_DIR}/")
