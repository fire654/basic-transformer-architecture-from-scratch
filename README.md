# Basic Transformer Architecture from Scratch

A minimal Transformer model implemented purely with PyTorch core functions, designed for text generation on the Shakespeare dataset.

## 📋 Overview
This project builds a Transformer from scratch (no high-level framework wrappers) and trains it on the Shakespeare dataset to generate Shakespeare-style text.

## 🚀 Quick Start
### 1. Train the Model
Run the training notebook to train the Transformer on the Shakespeare dataset:

`jupyter notebook train.ipynb`
### 2. Test the Model
Download the pre-trained model weights (state_dict) via the link below:
https://huggingface.co/fire654/shakespeare-transformer-generator/resolve/main/state_dict.pt?download=true  
Place the downloaded state_dict.pt in the project root directory before running test.ipynb.
Run the test notebook to generate text with the pre-trained model:


`jupyter notebook test.ipynb`
### 3. Core Code
Key functions (e.g., model layers, data processing) are implemented in:

`adapters.py`
