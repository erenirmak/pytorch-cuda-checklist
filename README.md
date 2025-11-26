# **🩺 PyTorch & CUDA Health Check**

A comprehensive, cross-platform **Jupyter Notebook** diagnostic tool to verify your Deep Learning environment.

It goes beyond simple availability checks to verify version integrity, hardware precision support (**Flash Attention**, **TF32**), and distributed training backends on both **Windows** and **Linux**.

## **🚀 Features**

This notebook performs a deep-dive scan of your system:

- **System vs. Bundled CUDA**: Compares the system-wide **nvcc** compiler against the CUDA runtime bundled with **PyTorch** to help debug path/version conflicts.

- **Attention Acceleration**: Verifies availability of **Scaled Dot Product Attention (SDPA)** and checks for specific backends:
  - **Flash Attention** (Fastest, requires specific hardware)
  - **Memory Efficient Attention**
  - **Math Fallback**

- **Hardware Capabilities**:
  - Translates **Compute Capability** numbers (e.g., 8.9) into usable insights.
  - Verifies support for **TensorFloat-32 (TF32)**, **BFloat16 (BF16)**, and **FP16** (Native vs. Storage).

- **Distributed Training Support**: Smart detection for all major backends, preventing crashes on Windows:
  - **NCCL** (Linux standard)
  - **Gloo** (Windows standard / Fallback)
  - **MPI**

- **Functional Stress Test**: Runs actual matrix multiplication on the GPU to catch "silent failure" driver issues.

- **Memory Diagnostics**: Reports real-time allocated vs. reserved GPU memory.

## **🛠️ Prerequisites**

- **Python 3.8** or higher  
- **Jupyter Notebook** or **JupyterLab**  
- **PyTorch** (installed via Pip or Conda)

## **📦 Quick Start**

Clone the repository:

```bash
git clone https://github.com/erenirmak/pytorch-cuda-checklist.git
cd pytorch-cuda-checklist
```

Install requirements:

```bash
pip install notebook torch
```

Run the diagnostics:

```bash
jupyter notebook torch-tests.ipynb
```

Run All Cells: Go to **Cell -> Run All** to generate your report.

## **🔍 Understanding the Output**

1. **Version Mismatches**

It is common for **System NVCC** to be different from **PyTorch CUDA**.

- **PyTorch CUDA**: Used for running standard models.  
- **System NVCC**: Only used if you are compiling custom extensions (like **flash-attn** from source).

2. **Flash Attention Support**

If you see **SDPA Available: Yes** but **Flash Attention: False**, it might be because:

- Your GPU is too old (**Flash Attention** usually requires Ampere / RTX 30-series or newer).  
- You are on **Windows** (support can sometimes be spotty for specific builds).  
- The datatype isn't supported (**fp16** or **bf16** often required).

3. **Distributed Backends**

- **Linux Users**: Expect **NCCL Available: Yes**.  
- **Windows Users**: Expect **NCCL Available: False** but **Gloo Available: True**.

## **📄 License**

This checklist is intended as a diagnostic aid. Adjust or extend tests for project-specific requirements.
