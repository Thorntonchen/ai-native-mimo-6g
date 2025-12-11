# AI-Native MIMO Transceiver: A Hybrid Neural-MMSE Approach

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📄 Abstract
This repository contains the official PyTorch implementation of the **"Grand Master" AI-Native Transceiver**. 

As 6G networks aim for extreme spectral efficiency, traditional physical layer (PHY) blocks struggle with non-linearities and complex channel dynamics. This project introduces a **Hybrid Neural Receiver** that integrates model-based linear MMSE equalization with a residual neural correction network. By utilizing a **Four-Phase Curriculum Learning** strategy, the system learns to communicate end-to-end, adapting modulation orders from BPSK up to **256-QAM** over a realistic, physics-compliant MIMO channel.

## 🌟 Key Features

*   **Hybrid "Grey Box" Receiver:** Combines the stability of linear MMSE with the non-linear learning capability of Deep Learning.
*   **Physics-Compliant Channel:** A custom `RealisticMIMOChannel` module that models thermal noise (-174 dBm/Hz), Path Loss, and Rician/Rayleigh fading.
*   **Gaussian Fourier Projection:** A Transformer-based decoder using Fourier features to resolve high-density constellations (spectral bias mitigation).
*   **RL Rate Manager:** An Actor-Critic agent that dynamically performs Adaptive Modulation and Coding (AMC) based on SNR and CSI.
*   **Curriculum Learning:** A 4-phase training loop ensuring convergence from cold-start to full joint optimization.

## 🚀 Usage
To start the training process:
```bash
python main.py

## 📂 Project Structure

```text
├── main.py       # Entry point: Runs the training loop and curriculum phases
├── models.py     # Architectures: Transmitter, Hybrid Receiver, Transformer Decoder
├── channel.py    # Physics: Realistic MIMO Channel implementation
├── training.py   # Curriculum logic: Loss functions and training steps
├── utils.py      # Helpers: Metrics, logging, and visualization tools
└── config.py     # Hyperparameters: Antennas, Bandwidth, Frequencies