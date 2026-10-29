#pragma once
#include "mnist_dataset.h"
#include <random>
#include <vector>

class Network {
public:
    Network(int in, int hid, int out, unsigned seed = 42);
    const float* forward(const unsigned char* image);
    // Uses activations cached by the most recent forward() call, so it must
    // be called right after forward() on the same sample.
    float backward(int label, float lr);
    // Trains on train_idx and reports accuracy on val_idx after each epoch.
    // Returns the final validation accuracy (%).
    float train(const MnistDataset& ds, std::vector<size_t> train_idx,
                const std::vector<size_t>& val_idx, int epochs, float lr);
    // Accuracy (%) on the given samples, without updating weights.
    float evaluate(const MnistDataset& ds, const std::vector<size_t>& idx);

private:
    int in_, hid_, out_;
    std::vector<float> W1_, b1_, W2_, b2_;

    std::mt19937 rng_;

    std::vector<float> x_, z1_, a1_, z2_, p_;
    std::vector<float> dz2_, da1_, dz1_;
};