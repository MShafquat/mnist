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

