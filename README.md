# Pytorch-Quantization-Example

This repository provides an example of Quantization-Aware Training (QAT) using the PyTorch framework, specifically applied to the MNIST dataset. It demonstrates how to prepare, train, and convert a neural network model for efficient deployment on hardware with limited computational resources.

## Getting Started

### Requirements

Please refer to [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) for instructions on installing PyTorch and torchvision.

### Installation

Clone this repository to your local machine:

```bash
git clone git@github.com:james397520/Pytorch-Quantization-Example.git
```

```bash
cd Pytorch-Quantization-Example
```

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

## Training mnist Model

### Training the Floating-Point Model

To train the floating-point model using the MNIST dataset:

```bash
python mnist_float.py
```

### Quantization-Aware Training (QAT)

```bash
cd QAT
```

#### 8-bit  QAT example:

```bash
python mnist_8bit.py
```

#### 4-bit  QAT example:

```bash
python mnist_4bit.py
```

## Test quantized model

```bash
python test_quantized_model.py
```


## Contributing

Contributions Welcome! Please open an issue or submit a pull request for any improvements or additions.