# cbbench
benchmarking tool
# cbbench3

**A lightweight CPU & GPU compute benchmark for tuning, performance testing, and throughput analysis.**

---

## Overview

`cbbench3` is a precision benchmarking tool designed to measure **raw CPU and GPU compute performance** using multithreaded CPU workloads and an ALU + memory stress GPU kernel.

It is **designed for enthusiasts, overclockers, and engineers** who want to:

- Evaluate **compute throughput** of CPU and GPU.
- Detect **thermal or power-limit throttling** during sustained workloads.
- Compare performance across **different hardware generations**.
- Fine-tune **CPU/GPU clocks, voltages, and power limits** for maximum stable performance.

Unlike traditional gaming benchmarks, `cbbench3` focuses on **controlled, repeatable compute workloads**, giving a clear and measurable metric for tuning.

---

## Features

- **CPU Benchmark**: Multithreaded workload across all cores, with a dynamic checksum to ensure work is actually performed.
- **GPU Benchmark**: OpenCL kernel stressing both ALU and memory, providing a realistic proxy for compute-heavy workloads.
- **Dynamic Checksums**: Ensures benchmarks are doing actual computation, not just looping empty data.
- **GPU Clock Readout**: Queries NVIDIA GPUs via `nvidia-smi` for real-time core and memory clocks.
- **Cross-Platform Ready**: Works on Windows (MinGW/MSYS) with OpenCL support; adaptable to Linux.
- **Lightweight & Fast**: Minimal dependencies, easy to compile and run.

---

## Installation

### Windows (MSYS2 / MinGW)

1. Install **OpenCL SDK** from your GPU vendor.
2. Install **MSYS2 / MinGW** with `g++`.
3. Compile:

```bash
g++ -std=c++17 -O2 -o cbbench3.exe cbbench3.cpp \
    -I"C:/msys64/mingw64/include" -L"C:/msys64/mingw64/lib" \
    -lgdi32 -luser32 -ld3d11 -ldxgi -ld3dcompiler -lole32 -lOpenCL
