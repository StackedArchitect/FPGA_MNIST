# FPGA Neural Network Accelerator for MNIST

This repository contains a neural network implementation on FPGA hardware using **Verilog**, supported by accompanying Python scripts for dataset preparation and memory file generation. The project loads pre-trained weights into the FPGA design using `.mem` files and performs inference for the **MNIST handwritten digit classification** dataset.

The goal of this project is to demonstrate how neural networks can be implemented on programmable logic, integrating hardware description, memory initialization, and testbench verification into a complete workflow.

## Features

- Verilog implementation of a neural network suitable for FPGA deployment
- Memory interface using `.mem` files for pre-trained weights and inputs
- Testbench for simulation and verification
- Python scripts to generate weight and input memory files
- Organized directory structure for hardware, verilog, and python assets

## Repository Structure

```
FPGA_NN/
│
├── verilog_files/           # Verilog source files for the neural network
├── python_files/            # Python scripts for generating memory data
├── diagrams/                # Diagrams illustrating architecture and design
├── MNIST/                   # MNIST dataset or references to dataset resources
│
├── *.mem                    # Memory files containing weights, biases, and input data
├── project_3.xpr            # Vivado project file (optional reference)
├── README.md                # This file
```

## Getting Started

To use this repository, follow the steps below:

### 1. Install Required Tools

- Install **Vivado Design Suite** (version compatible with your FPGA)
- Python 3.x installed on your machine
- Optional: ModelSim or other HDL simulator

Make sure your environment variables are setup for the Vivado tools and Python.

### 2. Prepare the Memory Files

Memory files in `.mem` format include weights, biases, and dataset inputs. These are used to initialize distributed memory or block RAM in the FPGA design.

The provided Python scripts in `python_files/` can be used to generate `.mem` files from trained network weights:

```
python generate_mem_files.py
```

Replace the filename as necessary based on your implementation.

### 3. Open the Vivado Project

Open the Vivado project file `project_3.xpr` using Vivado:

- Launch Vivado
- Select “Open Project”
- Choose `project_3.xpr`

The project contains HDL sources, constraints, memory initialization files, and testbenches.

### 4. Synthesize, Implement, and Generate Bitstream

Within Vivado:

- Run Synthesis
- Run Implementation
- Generate Bitstream

These steps target your FPGA board. Review any synthesis or timing reports for warnings or errors.

### 5. Simulate the Design

You can run simulation using Vivado or an external simulator:

- Launch simulation on the top-level testbench
- Observe waveform results
- Verify that outputs match expected labeled results from the `.mem` input files

### 6. Load onto FPGA (Optional)

After bitstream generation, you can program the FPGA board using hardware manager. Connect your board and click “Program Device”.

## Verilog Files Overview

The `verilog_files/` directory contains modules such as:

- `neural_network.sv` – top-level module
- `multiplier.sv` – multiplication unit
- `adder.sv` – addition unit
- `register.sv` – register and memory interface
- `ReLu.sv` – activation function
- `input_layer.sv`, `hidden_layer.sv`, `output_layer.sv` – layer definitions
- `tb_neuralnetwork.sv` – testbench

These modules implement the forward pass of the neural network in hardware logic.

## Python Files Overview

The `python_files/` directory contains scripts for:

- Reading the MNIST dataset
- Formatting data into `.mem` files
- Converting trained model weights into fixed-point representation
- Creating test input and expected output memory files

Review script comments for usage details.

## Notes

- `.mem` files are essential for weight initialization in hardware and should be included in the repository.
- Vivado auto-generated directories like `.cache/` and `.runs/` are ignored via `.gitignore`.
- This project is designed for educational and prototyping use and may require modification for different datasets or neural network architectures.

## Future Improvements

- Expand support for additional datasets and network architectures
- Integrate automated deployment scripts
- Add performance benchmarking and resource utilization reporting

## Author

**Adithya Ranjith**  
GitHub: https://github.com/toxx1n

If you find this project useful, please consider starring the repository.
