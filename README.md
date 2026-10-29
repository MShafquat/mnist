# MNIST in C++ (From Scratch)

A small neural network that learns to read handwritten digits, written in plain
C++ with no ML libraries.

## Get the data

```sh
./scripts/download_data.sh
```

This puts the four MNIST files in `data/`.

## Build

```sh
cmake -B build
cmake --build build
```

For a debug build with AddressSanitizer and UBSan:

```sh
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```

## Train

Run from the repo root (the data paths are relative):

```sh
./build/mnist
```

## Metrics

MLP (784 → 64 → 10, ReLU + softmax) trained with SGD, learning rate 0.01,
10 epochs, on an 80/20 train/val split of the 60k training set:

| Model     | Loss   | Train  | Val    | Test   |
|-----------|--------|--------|--------|--------|
| MLP + SGD | 0.0245 | 99.20% | 97.18% | 97.38% |

Test accuracy is measured on the separate 10k MNIST test set.