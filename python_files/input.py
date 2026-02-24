import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# ==========================================================================
# IMPORTANT: The transform MUST match what was used during training in
# model.py. The model was trained with Normalize((0.5,), (0.5,)) which
# maps pixels from [0,1] to [-1,1]. Using [0,1] here would cause wrong
# predictions because the weights were learned for [-1,1] range inputs.
# ==========================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # Match training: maps [0,1] -> [-1,1]
])

# Load MNIST test dataset
mnist_test = MNIST(root="./", train=False, download=True, transform=transform)

# ==========================================================================
# Choose which test image to use (change INDEX to pick a different digit)
# ==========================================================================
INDEX = 100  # Change this to test different images (0 to 9999)

image, label = mnist_test[INDEX]
image = image.squeeze().numpy()  # Convert to numpy array (28x28), range [-1, 1]

print(f"Selected test image index: {INDEX}")
print(f"Actual digit label: {label}")
print(f"Pixel value range: [{image.min():.4f}, {image.max():.4f}]")

# Fixed-point scaling (Q16.16 format)
# Values in [-1, 1] -> fixed-point [-65536, +65536]
scale_factor = 2**16
image_fixed = (image * scale_factor).astype(np.int32)

# Convert to 32-bit hexadecimal (two's complement for negatives)
hex_values = [format(int(val) & 0xFFFFFFFF, '08X') for row in image_fixed for val in row]

padding = ["00000000"] * 20
final_data = padding + hex_values + padding

print(f"Total data_in entries: {len(final_data)} (20 pad + 784 pixels + 20 pad)")

# Save input data to file
with open("data_in.mem", "w") as f:
    f.write("\n".join(final_data))

# Also save to parent directory for Vivado
with open("../data_in.mem", "w") as f:
    f.write("\n".join(final_data))

# Save the expected label to a separate file for testbench verification
with open("expected_label.mem", "w") as f:
    f.write(format(label, '08X'))
with open("../expected_label.mem", "w") as f:
    f.write(format(label, '08X'))

print(f"\ndata_in.mem generated successfully!")
print(f"expected_label.mem generated (contains label: {label})")
