# FPGA MNIST Digit Classifier

A hardware implementation of a neural network for handwritten digit classification (MNIST) on FPGA, written in SystemVerilog with a PyTorch software counterpart.

---

## Project Evolution

This project started as a **Multi-Layer Perceptron (MLP)** and was upgraded to a **1D Convolutional Neural Network (CNN)** to dramatically reduce the weight memory footprint — from 243,274 parameters down to 12,778 (19× reduction) while maintaining ~96.8% test accuracy.

---

## What's Inside

| Folder | Contents |
|--------|----------|
| `verilog_files/` | SystemVerilog RTL — conv1d, maxpool1d, FC layers, testbench |
| `python_files/` | PyTorch model training, weight export, and test utilities |
| `cnn_weights/` | Pre-exported `.mem` files ready for Vivado simulation |
| `diagrams/` | Block diagram PDFs for key modules |

---

## Quick Start

**Train and export weights:**
```bash
cd python_files
python cnn_model.py
```

**Test a specific MNIST image:**
```bash
python cnn_test_image.py 42
```

**Simulate in Vivado:**
1. Add all `verilog_files/*.sv` as sources, set `tb_cnn.sv` as sim top
2. Copy `cnn_weights/*.mem` to the Vivado sim working directory
3. Run: `run 200000ns`

For full implementation details see [`CNN_PROJECT_DOCUMENTATION.md`](CNN_PROJECT_DOCUMENTATION.md).

---

## Tech Stack

- **Hardware:** SystemVerilog, Xilinx Vivado XSim
- **Software:** Python 3, PyTorch
- **Arithmetic:** Q16.16 fixed-point throughout

---

## Roadmap

- [x] MLP baseline
- [x] 1D CNN upgrade
- [ ] **2D CNN implementation** — applying true 2D convolution directly on the 28×28 image grid for improved spatial feature extraction
