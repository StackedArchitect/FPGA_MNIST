import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import math

# Ternary Weight Network (TWN) + BatchNorm — 2D CNN for MNIST (FPGA-targeted)
#
#   Fixes Q16.16 overflow caused by unbounded activation accumulation in
#   pure {-1,0,+1} networks. BatchNorm keeps activations in a safe range
#   throughout the forward pass, bringing logits back into Q16.16 range.
#
#   Architecture:
#     Conv1  -> BN1 -> ReLU -> Pool1          [weights: {-1,0,+1}]
#     Conv2  -> BN2 -> ReLU -> Pool2          [weights: {-1,0,+1}]
#     Flatten
#     FC1    -> BN3 -> ReLU                   [weights: {-1,0,+1}]
#     FC2    -> logits                         [weights: {-1,0,+1}]

# ---- Architecture parameters ----
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

PAD                  = 20
FIXED_POINT_SCALE    = 2**16
TWN_THRESHOLD_FACTOR = 0.05
BN_EPS               = 1e-5     # matches PyTorch default


# TWN STE autograd function -- pure {-1, 0, +1}
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
        active_mask = (mask_pos | mask_neg).float()
        return grad_output * active_mask, None


# Ternary Conv2d and Linear
class TernaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        threshold = TWN_THRESHOLD_FACTOR * self.weight.detach().abs().max()
        q_weight  = TernaryQuantizeFn.apply(self.weight, threshold)
        return F.conv2d(x, q_weight, self.bias)

    @torch.no_grad()
    def get_ternary_info(self):
        threshold = TWN_THRESHOLD_FACTOR * self.weight.abs().max()
        mask_pos  = (self.weight >  threshold)
        mask_neg  = (self.weight < -threshold)
        codes     = mask_pos.int() - mask_neg.int()
        sparsity  = (codes == 0).float().mean().item()
        return codes, sparsity


class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        threshold = TWN_THRESHOLD_FACTOR * self.weight.detach().abs().max()
        q_weight  = TernaryQuantizeFn.apply(self.weight, threshold)
        return F.linear(x, q_weight, self.bias)

    @torch.no_grad()
    def get_ternary_info(self):
        threshold = TWN_THRESHOLD_FACTOR * self.weight.abs().max()
        mask_pos  = (self.weight >  threshold)
        mask_neg  = (self.weight < -threshold)
        codes     = mask_pos.int() - mask_neg.int()
        sparsity  = (codes == 0).float().mean().item()
        return codes, sparsity


# TWN + BatchNorm Model
class MNIST_CNN2D_TWN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TernaryConv2d(INPUT_CH,    CONV1_OUT_CH, CONV1_KERNEL)
        self.conv2 = TernaryConv2d(CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL)
        self.fc1   = TernaryLinear(FLATTEN_SIZE, FC1_OUT)
        self.fc2   = TernaryLinear(FC1_OUT, FC2_OUT)

        # BN placed BEFORE ReLU 
        # track_running_stats=True ensures running mean/var are updated
        # during training and frozen during eval -- critical for FPGA export
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
        return self.fc2(x)                        # (B, 10)

    def print_stats(self):
        print("\n  TWN+BN Weight Statistics (pure {-1, 0, +1}):")
        print(f"  {'Layer':<10} {'Shape':<20} {'Sparsity':>10} {'Active':>10}")
        print("  " + "-"*54)
        for name, layer in [('conv1', self.conv1), ('conv2', self.conv2),
                             ('fc1',   self.fc1),   ('fc2',   self.fc2)]:
            codes, sparsity = layer.get_ternary_info()
            n_total  = codes.numel()
            n_pos    = (codes ==  1).sum().item()
            n_neg    = (codes == -1).sum().item()
            n_active = n_pos + n_neg
            shape_str = str(tuple(layer.weight.shape))
            print(f"  {name:<10} {shape_str:<20} {sparsity*100:>9.1f}% "
                  f"{n_active:>5}/{n_total}  (+1:{n_pos}, -1:{n_neg})")

    def check_logit_range(self, test_loader, device, n_batches=10):
        """Verify logits stay within Q16.16 safe range [-32768, +32767]."""
        self.eval()
        max_logit = 0.0
        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                if i >= n_batches:
                    break
                logits    = self(images.to(device))
                max_logit = max(max_logit, logits.abs().max().item())
        q16_max = 32767.9999
        status  = "OK -- safe to proceed to hardware" if max_logit < q16_max \
                  else "OVERFLOW -- do NOT proceed to hardware"
        print(f"\n  Q16.16 range check:")
        print(f"    max |logit| observed : {max_logit:.4f}")
        print(f"    Q16.16 safe limit    : {q16_max:.0f}")
        print(f"    Status               : {status}")
        return max_logit < q16_max


# Warm-start
def warm_start(model, device):
    for ckpt in ["cnn2d_ttq_mnist_model.pth", "cnn2d_mnist_model.pth"]:
        if not os.path.exists(ckpt):
            continue
        src_sd = torch.load(ckpt, map_location=device)
        dst_sd = model.state_dict()
        loaded = 0
        for k in src_sd:
            if k in dst_sd and src_sd[k].shape == dst_sd[k].shape:
                dst_sd[k].copy_(src_sd[k])
                loaded += 1
        model.load_state_dict(dst_sd)
        print(f"[INFO] Warm-started from '{ckpt}' ({loaded} tensors loaded)")
        return
    print("[WARN] No checkpoint found. Training from scratch.")


# Training setup
print("=" * 60)
print("  TWN + BatchNorm 2D CNN -- MNIST")
print("=" * 60)
print(f"  Weights     : {{-1, 0, +1}} -- pure ternary, no scaling")
print(f"  BatchNorm   : after Conv1, Conv2, FC1 (before ReLU)")
print(f"  FPGA BN     : folded -> scale * x + shift per channel")
print(f"  Threshold   : Delta = {TWN_THRESHOLD_FACTOR} x max(|W_l|)")
print(f"  LR          : 1e-4 with grad clipping (max_norm=1.0)")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True,  transform=transform, download=True)
test_dataset  = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Device: {device}")

model     = MNIST_CNN2D_TWN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

warm_start(model, device)


# Training loop
EPOCHS = 15

print(f"\n[INFO] Training for {EPOCHS} epochs")
print(f"{'Epoch':<8} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 44)

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item()
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    scheduler.step()

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

    print(f"  {epoch+1:<6} {avg_loss:>12.4f} {train_acc:>9.2f}% {test_acc:>9.2f}%", end="")
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "cnn2d_twn_bn_mnist_model.pth")
        print("  <- best saved", end="")
    print()

print(f"\n[INFO] Best test accuracy: {best_acc:.2f}%")
print("[INFO] Model saved as cnn2d_twn_bn_mnist_model.pth")

model.load_state_dict(torch.load("cnn2d_twn_bn_mnist_model.pth", map_location=device))
model.eval()
model.print_stats()
model.check_logit_range(test_loader, device)


# Single-image verification (test[0])
test_image, test_label = test_dataset[0]
with torch.no_grad():
    logits     = model(test_image.unsqueeze(0).to(device))
    pred       = logits.argmax(dim=1).item()
    logits_np  = logits.cpu().numpy().flatten()
    q16_logits = (logits_np * FIXED_POINT_SCALE).astype(np.int64)

print(f"\n--- Single-image verification (test[0]) ---")
print(f"True label : {test_label}")
print(f"Predicted  : {pred}")
print(f"Logits     : {logits_np}")
print(f"Q16.16     : {q16_logits}")
q16_max = 2**31 - 1
if np.any(np.abs(q16_logits) > q16_max):
    print(">>> WARNING: int32 overflow in Q16.16 logits! <<<")
else:
    print(">>> Q16.16 logits within int32 range <<<")
if pred == test_label:
    print(">>> Software prediction: CORRECT <<<")
else:
    print(f">>> Software prediction: WRONG (expected {test_label}, got {pred}) <<<")


# Utility
def to_fixed_point_hex(value, scale=FIXED_POINT_SCALE):
    fixed = int(round(value * scale))
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF
    return format(fixed, '08X')

def to_ternary_2bit(code):
    """
    Encode a ternary value as a single 2-bit two's complement hex nibble.
      0  -> "0"   (2'b00)
     +1  -> "1"   (2'b01)
     -1  -> "3"   (2'b11)  <- two's complement of -1 in 2 bits
    Verilog: reg signed [1:0] mem [0:N-1];  $readmemh(...) reads this correctly.
    """
    return {0: "0", 1: "1", -1: "3"}[int(code)]


# ===========================================================================
# BatchNorm fold
#
#   Training BN:  y = gamma * (x - mean) / sqrt(var + eps) + beta
#   Folded form:  y = scale * x + shift
#     where:
#       scale = gamma / sqrt(var + eps)
#       shift = beta  - mean * scale
#
#   scale and shift are per-channel constants -- one multiply + one add
#   per output channel in RTL. Both exported as Q16.16.
# ===========================================================================
def get_folded_bn_params(bn_layer):
    gamma = bn_layer.weight.detach().cpu().numpy()
    beta  = bn_layer.bias.detach().cpu().numpy()
    mean  = bn_layer.running_mean.detach().cpu().numpy()
    var   = bn_layer.running_var.detach().cpu().numpy()
    eps   = bn_layer.eps
    scale = gamma / np.sqrt(var + eps)
    shift = beta  - mean * scale
    return scale, shift


# Export
CNN2D_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "weights")
os.makedirs(CNN2D_WEIGHTS_DIR, exist_ok=True)

def fp(name):
    return os.path.join(CNN2D_WEIGHTS_DIR, name)


def export_conv_with_bn(conv_layer, bn_layer, layer_name):
    codes, sparsity = conv_layer.get_ternary_info()
    w = codes.cpu().numpy()
    out_ch, in_ch, kH, kW = w.shape
    total = out_ch * in_ch * kH * kW

    code_lines = []
    for f in range(out_ch):
        for c in range(in_ch):
            for r in range(kH):
                for k in range(kW):
                    code_lines.append(to_ternary_2bit(int(w[f, c, r, k])))

    with open(fp(f"{layer_name}_ternary_codes.mem"), 'w') as f_:
        f_.write('\n'.join(code_lines))

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
    print(f"  {layer_name:<8} | {out_ch}x{in_ch}x{kH}x{kW} = {total:>4} weights | "
          f"sparsity={sparsity*100:.1f}%  (+1:{n_pos}, -1:{n_neg})")
    print(f"           BN scale [{scale.min():.4f}, {scale.max():.4f}]  "
          f"shift [{shift.min():.4f}, {shift.max():.4f}]")
    print(f"           -> {layer_name}_ternary_codes.mem, {layer_name}_b.mem, "
          f"{layer_name}_bn_scale.mem, {layer_name}_bn_shift.mem")


def export_fc_with_bn(fc_layer, bn_layer, layer_name, num_inputs):
    codes, sparsity = fc_layer.get_ternary_info()
    w = codes.cpu().numpy()
    num_neurons = w.shape[0]
    padding     = ["0"] * PAD   # 2-bit zero padding (was "00000000" for 32-bit)

    code_lines = []
    for n in range(num_neurons):
        row = [to_ternary_2bit(int(v)) for v in w[n]]
        code_lines.extend(padding + row + padding)

    with open(fp(f"{layer_name}_ternary_codes.mem"), 'w') as f_:
        f_.write('\n'.join(code_lines))

    b = fc_layer.bias.detach().cpu().numpy()
    with open(fp(f"{layer_name}_b.mem"), 'w') as f_:
        f_.write('\n'.join(to_fixed_point_hex(v) for v in b))

    n_pos = (codes == 1).sum().item()
    n_neg = (codes == -1).sum().item()
    total = num_neurons * num_inputs
    print(f"  {layer_name:<8} | {num_neurons}x{num_inputs} = {total:>5} weights | "
          f"sparsity={sparsity*100:.1f}%  (+1:{n_pos}, -1:{n_neg})")

    if bn_layer is not None:
        scale, shift = get_folded_bn_params(bn_layer)
        with open(fp(f"{layer_name}_bn_scale.mem"), 'w') as f_:
            f_.write('\n'.join(to_fixed_point_hex(v) for v in scale))
        with open(fp(f"{layer_name}_bn_shift.mem"), 'w') as f_:
            f_.write('\n'.join(to_fixed_point_hex(v) for v in shift))
        print(f"           BN scale [{scale.min():.4f}, {scale.max():.4f}]  "
              f"shift [{shift.min():.4f}, {shift.max():.4f}]")
        print(f"           -> {layer_name}_ternary_codes.mem, {layer_name}_b.mem, "
              f"{layer_name}_bn_scale.mem, {layer_name}_bn_shift.mem")
    else:
        print(f"           -> {layer_name}_ternary_codes.mem, {layer_name}_b.mem  (no BN)")


# ---- Run export ----
model_cpu = model.to('cpu')
model_cpu.eval()

print(f"\n--- Exporting TWN+BN weights to {CNN2D_WEIGHTS_DIR}/ ---")
print(f"  Weights  : 2-bit ternary codes (0='0', +1='1', -1='3')")
print(f"  Verilog  : reg signed [1:0] mem [0:N-1];  $readmemh(...)")
print(f"  BN       : folded -> scale * x + shift (Q16.16 per channel)")
print(f"  {'Layer':<8}   Details")
print("  " + "-"*70)

export_conv_with_bn(model_cpu.conv1, model_cpu.bn1, "conv1")
export_conv_with_bn(model_cpu.conv2, model_cpu.bn2, "conv2")
export_fc_with_bn(model_cpu.fc1, model_cpu.bn3, "fc1", FLATTEN_SIZE)
export_fc_with_bn(model_cpu.fc2, None,          "fc2", FC1_OUT)

# ---- Test input ----
print(f"\n--- Generating test input (test[0]) ---")
image_np   = test_dataset[0][0].squeeze().numpy()
hex_pixels = [to_fixed_point_hex(image_np[r, c])
              for r in range(INPUT_H) for c in range(INPUT_W)]
with open(fp("data_in.mem"), 'w') as f_:
    f_.write('\n'.join(hex_pixels))
with open(fp("expected_label.mem"), 'w') as f_:
    f_.write(format(test_dataset[0][1], '08X'))

print(f"  data_in.mem        ({len(hex_pixels)} pixels, row-major 28x28)")
print(f"  expected_label.mem (label={test_dataset[0][1]})")
print(f"\n[DONE] All .mem files written to cnn2d_weights/")
