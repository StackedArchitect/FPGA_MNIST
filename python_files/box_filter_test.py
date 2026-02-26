import numpy as np
import os

# ===========================================================================
# Box Filter Test for conv2d.sv
#
# Creates a simple, hand-verifiable test case:
#   - 6×6 input, 1 channel, values 1..36 (row-major)
#   - 2 output filters, each is a 3×3 box filter (all weights = 1.0)
#   - Bias = 0 for filter 0, bias = 10.0 for filter 1
#   - No activation (activation_function = 0) so we see raw sums
#
# With weight=1.0 everywhere, the output at each position is the sum
# of the 9 values in the 3×3 neighborhood, making it trivial to verify.
#
# For filter 0 (bias=0), output[r][c] = sum of 3×3 block starting at (r,c)
# For filter 1 (bias=10), output[r][c] = same sum + 10.0
#
# Input (6×6):
#    1   2   3   4   5   6
#    7   8   9  10  11  12
#   13  14  15  16  17  18
#   19  20  21  22  23  24
#   25  26  27  28  29  30
#   31  32  33  34  35  36
#
# Output (4×4 per filter, valid convolution):
#   Filter 0 (bias=0):
#     pos(0,0): 1+2+3+7+8+9+13+14+15              = 72
#     pos(0,1): 2+3+4+8+9+10+14+15+16             = 81
#     pos(0,2): 3+4+5+9+10+11+15+16+17            = 90
#     pos(0,3): 4+5+6+10+11+12+16+17+18           = 99
#     pos(1,0): 7+8+9+13+14+15+19+20+21           = 126
#     ... etc
#
#   Filter 1 = Filter 0 + 10.0
# ===========================================================================

FIXED_POINT_SCALE = 2**16   # Q16.16

# ---- Test parameters ----
IN_H     = 6
IN_W     = 6
IN_CH    = 1
OUT_CH   = 2
KERNEL_H = 3
KERNEL_W = 3
OUT_H    = IN_H - KERNEL_H + 1  # 4
OUT_W    = IN_W - KERNEL_W + 1  # 4

def to_q16_hex(value):
    """Convert float to Q16.16 hex string (32-bit, two's complement)."""
    fixed = int(round(value * FIXED_POINT_SCALE))
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF
    return format(fixed, '08X')


# ---- Build input: 1, 2, 3, ..., 36 ----
input_data = np.arange(1, IN_H * IN_W + 1, dtype=np.float64).reshape(IN_CH, IN_H, IN_W)
print("Input (6×6):")
print(input_data[0])

# ---- Build box filter weights: all 1.0 ----
# Shape: (OUT_CH, IN_CH, KERNEL_H, KERNEL_W)
weights = np.ones((OUT_CH, IN_CH, KERNEL_H, KERNEL_W), dtype=np.float64)

# ---- Biases ----
biases = np.array([0.0, 10.0], dtype=np.float64)

# ---- Compute expected output in Python ----
print(f"\nExpected output ({OUT_H}×{OUT_W} per filter):")
expected = np.zeros((OUT_CH, OUT_H, OUT_W), dtype=np.float64)
for f in range(OUT_CH):
    for r in range(OUT_H):
        for c in range(OUT_W):
            acc = biases[f]
            for ch in range(IN_CH):
                for kr in range(KERNEL_H):
                    for kc in range(KERNEL_W):
                        acc += input_data[ch, r + kr, c + kc] * weights[f, ch, kr, kc]
            expected[f, r, c] = acc

for f in range(OUT_CH):
    print(f"\nFilter {f} (bias={biases[f]}):")
    print(expected[f])

# ---- Also show Q16.16 values ----
print(f"\nQ16.16 integer values (multiply by 65536):")
for f in range(OUT_CH):
    print(f"\nFilter {f}:")
    for r in range(OUT_H):
        row_vals = [f"{int(expected[f, r, c] * FIXED_POINT_SCALE):>12d}" for c in range(OUT_W)]
        print("  " + "  ".join(row_vals))

# ---- Export .mem files ----
BOX_DIR = os.path.join(os.path.dirname(__file__), "..", "box_filter_test")
os.makedirs(BOX_DIR, exist_ok=True)

# Input data
with open(os.path.join(BOX_DIR, "box_data_in.mem"), 'w') as f:
    for ch in range(IN_CH):
        for r in range(IN_H):
            for c in range(IN_W):
                f.write(to_q16_hex(input_data[ch, r, c]) + '\n')
print(f"\nGenerated box_data_in.mem  ({IN_H * IN_W * IN_CH} entries)")

# Weights: flat [f][ch][kr][kc]
with open(os.path.join(BOX_DIR, "box_weights.mem"), 'w') as f:
    for filt in range(OUT_CH):
        for ch in range(IN_CH):
            for kr in range(KERNEL_H):
                for kc in range(KERNEL_W):
                    f.write(to_q16_hex(weights[filt, ch, kr, kc]) + '\n')
print(f"Generated box_weights.mem  ({OUT_CH * IN_CH * KERNEL_H * KERNEL_W} entries)")

# Biases
with open(os.path.join(BOX_DIR, "box_bias.mem"), 'w') as f:
    for filt in range(OUT_CH):
        f.write(to_q16_hex(biases[filt]) + '\n')
print(f"Generated box_bias.mem     ({OUT_CH} entries)")

# Expected output (for reference / automatic checking in testbench)
with open(os.path.join(BOX_DIR, "box_expected.mem"), 'w') as f:
    for filt in range(OUT_CH):
        for r in range(OUT_H):
            for c in range(OUT_W):
                f.write(to_q16_hex(expected[filt, r, c]) + '\n')
print(f"Generated box_expected.mem ({OUT_CH * OUT_H * OUT_W} entries)")

print(f"\nAll files in: {BOX_DIR}/")
print(f"Parameters for testbench:")
print(f"  IN_H={IN_H}, IN_W={IN_W}, IN_CH={IN_CH}")
print(f"  OUT_CH={OUT_CH}, KERNEL_H={KERNEL_H}, KERNEL_W={KERNEL_W}")
print(f"  OUT_H={OUT_H}, OUT_W={OUT_W}")
print(f"  Total output values: {OUT_CH * OUT_H * OUT_W}")
