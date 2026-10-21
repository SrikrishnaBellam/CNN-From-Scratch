# CNN from Scratch (NumPy only)

A convolutional neural network implemented **without PyTorch, TensorFlow or any deep learning library**: every layer, forward pass and gradient is written by hand in NumPy. The goal is to understand exactly what a framework does when you call `loss.backward()`.

## What's implemented

| Component | Details |
|---|---|
| `Conv_Layer` | Learnable filters and biases; forward convolution, plus a `gradient_descent_step` that backpropagates and updates them |
| `Pooling` | Max (or min) pooling, passing gradients back to the winning positions |
| `Flatten` | Reshapes feature maps for the dense head |
| `FullConnectedNeuralNetwork` | Dense layers with forward and backward passes |
| Activations | Sigmoid, ReLU, Leaky ReLU and tanh, each with its derivative |
| Output and loss | Softmax with cross-entropy loss and its derivative |
| Training | Mini-batch SGD with a custom `DataBatcher` (batch size 64, shuffling) |
| Safety | A `NaNException` that stops training if gradients blow up |

`ConvolutionalNeuralNetwork.train()` ties it together: a forward pass through every layer, softmax, then each layer's `gradient_descent_step` in reverse order to backpropagate the loss and update weights, with validation accuracy tracked during training.

## Dataset

The notebook trains on handwritten digits from Kaggle's **Digit Recognizer** (MNIST) `train.csv`. The same code also runs on FashionMNIST.

## Repository structure

```
CNNFromScratch.ipynb   # layers, training loop and evaluation
```

## Running it

Download `train.csv` from the Kaggle Digit Recognizer competition, update the path in the data-loading cell (it was written for Google Colab with Drive mounted), and run the notebook.

**Requirements:** Python 3, NumPy, pandas, SciPy, scikit-image, Matplotlib, tqdm.

## Tech stack

`NumPy` · `Convolutional Neural Networks` · `Backpropagation` · `SGD` · `From Scratch`
