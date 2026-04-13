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

# ===========================================================================
# Trained Ternary Quantization (TTQ) + BatchNorm
# 2D CNN for MNIST  (FPGA-targeted, Zynq-7020)
#
# Reference: Zhu et al., "Trained Ternary Quantization", ICLR 2017
#
# Method:
#   Each layer l has two learnable positive scalars Wp_l and Wn_l.
#   Forward: W_t = { +Wp  if W~ >  Delta
#                  {    0  if |W~| <= Delta
#                  { -Wn  if W~ < -Delta
#   where Delta = 0.05 * max(|W~|)   [paper eq. 9]
#
#   Backward (paper eq. 7-8):
#     grad_Wp  = sum of upstream grads over positive-masked positions
#     grad_Wn  = -sum of upstream grads over negative-masked positions
#     grad_W~  = Wp * upstream_grad   (positive positions)
#              =  1 * upstream_grad   (zero positions)
#              = Wn * upstream_grad   (negative positions)
#
#   Without BN, TTQ logits reached ~374 in our earlier run — near the
#   Q16.16 limit of 32767. BN keeps activations bounded throughout the
#   forward pass, preventing overflow on hardware.
# #
# Architecture:
#   Conv1 -> BN1 -> ReLU -> Pool1      [weights: {-Wn, 0, +Wp} per layer]
#   Conv2 -> BN2 -> ReLU -> Pool2
#   Flatten
#   FC1   -> BN3 -> ReLU
#   FC2   -> logits                    [no BN on output layer]

# FPGA inference pipeline per layer:
#   1. Split MAC:
#        pos_acc = sum of activations where code == +1
#        neg_acc = sum of activations where code == -1
#        acc     = Wp * pos_acc - Wn * neg_acc  + bias
#   2. Folded BN:
#        out = bn_scale * acc + bn_shift
#   3. ReLU:  max(0, out)
#   FC2: no BN -- MAC + bias + argmax
# ===========================================================================

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
FIXED_POINT_SCALE    = 2**16        # Q16.16
TTQ_THRESHOLD_FACTOR = 0.05         # Delta = 0.05 * max(|W~|)  [paper eq. 9]
BN_EPS               = 1e-5


# ===========================================================================
# TTQ STE autograd function  (paper eq. 6, 7, 8)
#
# Forward:
#   W_t[i] = +Wp   if  W~[i] >  Delta
#   W_t[i] =   0   if |W~[i]| <= Delta
#   W_t[i] = -Wn   if  W~[i] < -Delta
#
# Backward:
#   grad_W~[i] = Wp * grad_out[i]   if mask_pos[i]   (paper eq. 8)
#   grad_W~[i] =  1 * grad_out[i]   if zero band
#   grad_W~[i] = Wn * grad_out[i]   if mask_neg[i]
#   grad_Wp    = sum(grad_out[i] for i in mask_pos)   (paper eq. 7)
#   grad_Wn    = -sum(grad_out[i] for i in mask_neg)  (paper eq. 7)
# ===========================================================================
class TTQQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, wp, wn, threshold):
        """
        weight    : latent full-precision weights  (any shape)
        wp        : scalar positive scaling factor (shape [1], positive)
        wn        : scalar negative scaling factor (shape [1], positive)
        threshold : scalar Delta = 0.05 * max(|weight|)
        Returns   : quantised weight tensor {+wp, 0, -wn}
        """
        mask_pos = (weight >  threshold)
        mask_neg = (weight < -threshold)
        # {+Wp, 0, -Wn}
        q_weight = mask_pos.float() * wp - mask_neg.float() * wn
        # Save masks and scaling factors for backward
        ctx.save_for_backward(mask_pos, mask_neg, wp, wn)
        return q_weight

    @staticmethod
    def backward(ctx, grad_output):
        mask_pos, mask_neg, wp, wn = ctx.saved_tensors
        mask_zero = ~(mask_pos | mask_neg)

        # grad_W~ per paper eq. 8:
        #   Wp * grad  on positive positions
        #    1 * grad  on zero positions
        #   Wn * grad  on negative positions
        grad_weight = (mask_pos.float() * wp
                     + mask_zero.float() * 1.0
                     + mask_neg.float() * wn) * grad_output

        # grad_Wp and grad_Wn per paper eq. 7
        # .view(1) ensures shape matches nn.Parameter(torch.ones(1))
        grad_wp = (grad_output * mask_pos.float()).sum().view(1)
        grad_wn = -(grad_output * mask_neg.float()).sum().view(1)

        return grad_weight, grad_wp, grad_wn, None  # None for threshold


# TTQ Conv2d layer
class TTQConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size

        # Latent full-precision weights -- updated by Adam, not the ternary codes
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Learnable per-layer scaling factors (paper eq. 6)
        # Initialised to mean(|W|) after warm-start, or 0.1 from scratch
        self.wp = nn.Parameter(torch.ones(1) * 0.1)
        self.wn = nn.Parameter(torch.ones(1) * 0.1)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Threshold is computed from current latent weights (paper eq. 9)
        # .detach() prevents max from entering the autograd graph
        threshold = TTQ_THRESHOLD_FACTOR * self.weight.detach().abs().max()
        wp = self.wp.abs()   # enforce positivity at all times
        wn = self.wn.abs()
        q_weight = TTQQuantizeFn.apply(self.weight, wp, wn, threshold)
        return F.conv2d(x, q_weight, self.bias)

    @torch.no_grad()
    def get_ternary_info(self):
        """
        Returns codes {-1, 0, +1}, wp_val, wn_val, quantised weights, sparsity.
        Called only at export time -- not during training.
        """
        threshold = TTQ_THRESHOLD_FACTOR * self.weight.abs().max()
        wp_val    = self.wp.abs().item()
        wn_val    = self.wn.abs().item()
        mask_pos  = (self.weight >  threshold)
        mask_neg  = (self.weight < -threshold)
        codes     = mask_pos.int() - mask_neg.int()     # {-1, 0, +1}
        q_weights = mask_pos.float() * wp_val - mask_neg.float() * wn_val
        sparsity  = (codes == 0).float().mean().item()
        return codes, wp_val, wn_val, q_weights, sparsity


# TTQ Linear layer
class TTQLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.wp     = nn.Parameter(torch.ones(1) * 0.1)
        self.wn     = nn.Parameter(torch.ones(1) * 0.1)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        threshold = TTQ_THRESHOLD_FACTOR * self.weight.detach().abs().max()
        wp = self.wp.abs()
        wn = self.wn.abs()
        q_weight = TTQQuantizeFn.apply(self.weight, wp, wn, threshold)
        return F.linear(x, q_weight, self.bias)

    @torch.no_grad()
    def get_ternary_info(self):
        threshold = TTQ_THRESHOLD_FACTOR * self.weight.abs().max()
        wp_val    = self.wp.abs().item()
        wn_val    = self.wn.abs().item()
        mask_pos  = (self.weight >  threshold)
        mask_neg  = (self.weight < -threshold)
        codes     = mask_pos.int() - mask_neg.int()
        q_weights = mask_pos.float() * wp_val - mask_neg.float() * wn_val
        sparsity  = (codes == 0).float().mean().item()
        return codes, wp_val, wn_val, q_weights, sparsity


class MNIST_CNN2D_TTQ_BN(nn.Module):
    def __init__(self):
        super().__init__()
        # TTQ conv and FC layers
        self.conv1 = TTQConv2d(INPUT_CH,    CONV1_OUT_CH, CONV1_KERNEL)
        self.conv2 = TTQConv2d(CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL)
        self.fc1   = TTQLinear(FLATTEN_SIZE, FC1_OUT)
        self.fc2   = TTQLinear(FC1_OUT, FC2_OUT)

        # BatchNorm -- placed before ReLU (standard for quantised networks)
        # track_running_stats=True ensures running mean/var are accumulated
        # during training and frozen during eval, ready for folded export
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

    def print_stats(self):
        print("\n  TTQ+BN Weight Statistics:")
        print(f"  {'Layer':<10} {'Shape':<22} {'Wp':>8} {'Wn':>8} "
              f"{'Sparsity':>10} {'Active':>10}")
        print("  " + "-"*72)
        for name, layer in [('conv1', self.conv1), ('conv2', self.conv2),
                             ('fc1',   self.fc1),   ('fc2',   self.fc2)]:
            codes, wp, wn, _, sparsity = layer.get_ternary_info()
            n_total  = codes.numel()
            n_pos    = (codes ==  1).sum().item()
            n_neg    = (codes == -1).sum().item()
            n_active = n_pos + n_neg
            shape_str = str(tuple(layer.weight.shape))
            print(f"  {name:<10} {shape_str:<22} {wp:>8.5f} {wn:>8.5f} "
                  f"{sparsity*100:>9.1f}% {n_active:>5}/{n_total}"
                  f"  (+1:{n_pos}, -1:{n_neg})")

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


def warm_start(model, device):
    priority = [
        "cnn2d_twn_bn_mnist_model.pth",
        "cnn2d_ttq_mnist_model.pth",
        "cnn2d_mnist_model.pth",
    ]
    for ckpt in priority:
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
        print(f"[INFO] Warm-started from '{ckpt}'  ({loaded} tensors loaded)")

        # Re-initialise wp, wn to mean(|W|) per layer after weight loading.
        # This gives TTQ a sensible initial scale for the positive/negative
        # separation, rather than the 0.1 default.
        for layer in [model.conv1, model.conv2, model.fc1, model.fc2]:
            mean_abs = layer.weight.data.abs().mean().item()
            layer.wp.data.fill_(mean_abs)
            layer.wn.data.fill_(mean_abs)
        print(f"[INFO] wp, wn re-initialised to mean(|W|) per layer")
        return
    print("[WARN] No checkpoint found. Training from scratch.")


# ===========================================================================
# Training setup
# ===========================================================================
print("=" * 60)
print("  TTQ + BN 2D CNN -- MNIST")
print("=" * 60)
print(f"  Quantisation : TTQ  W_t in {{-Wn, 0, +Wp}}  [paper eq. 6]")
print(f"  Threshold    : Delta = {TTQ_THRESHOLD_FACTOR} x max(|W~_l|)  [paper eq. 9]")
print(f"  Backward     : scaled STE per paper eq. 7-8")
print(f"  BatchNorm    : after Conv1, Conv2, FC1  (our addition for FPGA)")
print(f"  Biases       : full precision")
print(f"  Layers       : conv1, conv2, fc1, fc2  (all ternarised)")
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

model     = MNIST_CNN2D_TTQ_BN().to(device)
criterion = nn.CrossEntropyLoss()

# Separate LR groups:
#   - scaling factors (wp, wn): same LR as weights -- paper trains them jointly
#   - other params (weights, BN): base LR
scaling_params = [p for n, p in model.named_parameters()
                  if n.endswith('.wp') or n.endswith('.wn')]
other_params   = [p for n, p in model.named_parameters()
                  if not (n.endswith('.wp') or n.endswith('.wn'))]

optimizer = optim.Adam([
    {'params': other_params,   'lr': 1e-3},   # slightly higher for weights
    {'params': scaling_params, 'lr': 1e-3},   # same LR for Wp/Wn
], weight_decay=1e-4)

# Gentler scheduler than original TTQ run (step=5) -- BN stabilises training
# so we can afford a slower decay and gain better final accuracy
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

warm_start(model, device)


# ===========================================================================
# Training loop
# ===========================================================================
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

        # Gradient clipping prevents large batches from flipping ternary
        # assignments en masse in a single step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Clamp wp, wn to remain strictly positive after each optimizer step.
        # Without this, Adam can push them negative, which flips the sign
        # convention and causes training instability.
        with torch.no_grad():
            for layer in [model.conv1, model.conv2, model.fc1, model.fc2]:
                layer.wp.data.clamp_(min=1e-4)
                layer.wn.data.clamp_(min=1e-4)

        total_loss    += loss.item()
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    scheduler.step()

    # ---- Evaluation: model.eval() switches BN to running stats mode ----
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
        torch.save(model.state_dict(), "cnn2d_ttq_bn_mnist_model.pth")
        print("  <- best saved", end="")
    print()

print(f"\n[INFO] Best test accuracy: {best_acc:.2f}%")
print("[INFO] Model saved as cnn2d_ttq_bn_mnist_model.pth")

# ---- Load best checkpoint and report ----
model.load_state_dict(
    torch.load("cnn2d_ttq_bn_mnist_model.pth", map_location=device))
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
if np.any(np.abs(q16_logits) > 2**31 - 1):
    print(">>> WARNING: int32 overflow in Q16.16 logits! <<<")
else:
    print(">>> Q16.16 logits within int32 range <<<")
if pred == test_label:
    print(">>> Software prediction: CORRECT <<<")
else:
    print(f">>> Software prediction: WRONG (expected {test_label}, got {pred}) <<<")


# Utility functions

def to_fixed_point_hex(value, scale=FIXED_POINT_SCALE):
    """Convert float to Q16.16 32-bit two's complement hex string."""
    fixed = int(round(value * scale))
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF
    return format(fixed, '08X')

def to_ternary_2bit(code):
    return {0: "0", 1: "1", -1: "3"}[int(code)]


# ===========================================================================
# BatchNorm fold -- for FPGA inference
#
# Training BN:  y = gamma * (x - mean) / sqrt(var + eps) + beta
# Folded form:  y = scale * x + shift
#   scale = gamma / sqrt(var + eps)
#   shift = beta  - mean * scale
#
# One multiply + one add per channel in RTL.
# Running stats are only valid after model.eval() -- always call eval()
# before exporting.
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


# Export directory
CNN2D_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "weights")
os.makedirs(CNN2D_WEIGHTS_DIR, exist_ok=True)

def fp(name):
    return os.path.join(CNN2D_WEIGHTS_DIR, name)


# Export functions
def export_ttq_conv_with_bn(conv_layer, bn_layer, layer_name):
    """
    Traversal order: (out_ch, in_ch, kH, kW)
    Matches RTL: filter_idx * TAP_COUNT + ch_cnt * KH * KW + kr * KW + kc
    """
    codes, wp_val, wn_val, _, sparsity = conv_layer.get_ternary_info()
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

    # Scalar Wp and Wn (one value per layer, not per filter)
    with open(fp(f"{layer_name}_wp.mem"), 'w') as f_:
        f_.write(to_fixed_point_hex(wp_val))
    with open(fp(f"{layer_name}_wn.mem"), 'w') as f_:
        f_.write(to_fixed_point_hex(wn_val))

    # Per-filter bias
    b = conv_layer.bias.detach().cpu().numpy()
    with open(fp(f"{layer_name}_b.mem"), 'w') as f_:
        f_.write('\n'.join(to_fixed_point_hex(v) for v in b))

    # Folded BN
    scale, shift = get_folded_bn_params(bn_layer)
    with open(fp(f"{layer_name}_bn_scale.mem"), 'w') as f_:
        f_.write('\n'.join(to_fixed_point_hex(v) for v in scale))
    with open(fp(f"{layer_name}_bn_shift.mem"), 'w') as f_:
        f_.write('\n'.join(to_fixed_point_hex(v) for v in shift))

    n_pos = (codes == 1).sum().item()
    n_neg = (codes == -1).sum().item()
    print(f"  {layer_name:<8} | {out_ch}x{in_ch}x{kH}x{kW} = {total:>4} weights | "
          f"Wp={wp_val:.5f}  Wn={wn_val:.5f}  sparsity={sparsity*100:.1f}%"
          f"  (+1:{n_pos}, -1:{n_neg})")
    print(f"           BN scale [{scale.min():.4f}, {scale.max():.4f}]  "
          f"shift [{shift.min():.4f}, {shift.max():.4f}]")
    print(f"           -> {layer_name}_ternary_codes.mem, _wp.mem, _wn.mem, "
          f"_b.mem, _bn_scale.mem, _bn_shift.mem")


def export_ttq_fc_with_bn(fc_layer, bn_layer, layer_name, num_inputs):
    """
    Exports for a TTQLinear layer.
    PAD zero entries on each side (legacy compatibility with float FC export).
    If bn_layer is None (fc2), only codes + Wp/Wn + bias are exported.
    """
    codes, wp_val, wn_val, _, sparsity = fc_layer.get_ternary_info()
    w = codes.cpu().numpy()
    num_neurons = w.shape[0]
    total       = num_neurons * num_inputs
    padding     = ["0"] * PAD   # 2-bit zero padding

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

    n_pos = (codes == 1).sum().item()
    n_neg = (codes == -1).sum().item()
    print(f"  {layer_name:<8} | {num_neurons}x{num_inputs} = {total:>5} weights | "
          f"Wp={wp_val:.5f}  Wn={wn_val:.5f}  sparsity={sparsity*100:.1f}%"
          f"  (+1:{n_pos}, -1:{n_neg})")

    if bn_layer is not None:
        scale, shift = get_folded_bn_params(bn_layer)
        with open(fp(f"{layer_name}_bn_scale.mem"), 'w') as f_:
            f_.write('\n'.join(to_fixed_point_hex(v) for v in scale))
        with open(fp(f"{layer_name}_bn_shift.mem"), 'w') as f_:
            f_.write('\n'.join(to_fixed_point_hex(v) for v in shift))
        print(f"           BN scale [{scale.min():.4f}, {scale.max():.4f}]  "
              f"shift [{shift.min():.4f}, {shift.max():.4f}]")
        print(f"           -> {layer_name}_ternary_codes.mem, _wp.mem, _wn.mem, "
              f"_b.mem, _bn_scale.mem, _bn_shift.mem")
    else:
        print(f"           -> {layer_name}_ternary_codes.mem, _wp.mem, _wn.mem, "
              f"_b.mem  (no BN on output layer)")

# Run full export
model_cpu = model.to('cpu')
model_cpu.eval()   # CRITICAL: running stats must be frozen before export

print(f"\n--- Exporting TTQ+BN weights to {CNN2D_WEIGHTS_DIR}/ ---")
print(f"  Ternary codes : 2-bit  (1=+1, 3=-1, 0=0)")
print(f"  Wp / Wn       : Q16.16 scalar per layer")
print(f"  BN            : folded -> scale * x + shift  (Q16.16 per channel)")
print(f"  {'Layer':<8}   Details")
print("  " + "-"*72)

export_ttq_conv_with_bn(model_cpu.conv1, model_cpu.bn1, "conv1")
export_ttq_conv_with_bn(model_cpu.conv2, model_cpu.bn2, "conv2")
export_ttq_fc_with_bn(model_cpu.fc1, model_cpu.bn3, "fc1", FLATTEN_SIZE)
export_ttq_fc_with_bn(model_cpu.fc2, None,           "fc2", FC1_OUT)

# ---- Test input (identical format to previous models) ----
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
print(f"\n[DONE] All .mem files written to weights")
