#include "network.h"
#include "matmul.h"
#include "mnist_dataset.h"
#include <algorithm>
#include <cmath>
#include <print>

// Weight matrices are row-major: W1 is hid×in, W2 is out×hid.
Network::Network(int in, int hid, int out, unsigned seed)
    : in_(in), hid_(hid), out_(out),
    W1_(hid * in), b1_(hid, 0.0f),
    W2_(out * hid), b2_(out, 0.0f),
    rng_(seed),
    x_(in), z1_(hid), a1_(hid), z2_(out), p_(out),
    dz2_(out), da1_(hid), dz1_(hid) {
        // He init (stddev = sqrt(2 / fan_in)) keeps activations from shrinking
        // or blowing up layer to layer with ReLU. Biases start at zero.
        std::normal_distribution<float> nd1(0.0f, std::sqrt(2.0f / in));
        std::normal_distribution<float> nd2(0.0f, std::sqrt(2.0f / hid));
        for (auto& w: W1_) w = nd1(rng_);
        for (auto& w: W2_) w = nd2(rng_);
}

const float* Network::forward(const unsigned char* image) {
    // Scale pixels from [0, 255] to [0, 1].
    for (int i = 0; i < in_; i++) {
        x_[i] = image[i] / 255.0f;
    }

    // Hidden layer: z1 = W1·x + b1, a1 = ReLU(z1)
    matvec(W1_.data(), x_.data(), z1_.data(), hid_, in_);
    for (int j = 0; j < hid_; j++) z1_[j] += b1_[j];
    for (int j = 0; j < hid_; j++) a1_[j] = z1_[j] > 0.0f ? z1_[j] : 0.0f;

    // Output layer: z2 = W2·a1 + b2
    matvec(W2_.data(), a1_.data(), z2_.data(), out_, hid_);
    for (int k = 0; k < out_; k++) z2_[k] += b2_[k];

    // Softmax. Subtracting the max logit doesn't change the result but
    // keeps exp() from overflowing.
    float m = *std::max_element(z2_.begin(), z2_.end());
    float s = 0;
    for (int k = 0; k < out_; k++) {
        p_[k] = std::exp(z2_[k] - m);
        s += p_[k];
    }
    for (int k = 0; k < out_; k++) {
        p_[k] /= s;
    }

    return p_.data();
}

float Network::backward(int label, float lr) {
    // Cross-entropy loss, only returned for logging. The clamp avoids log(0).
    float loss = -std::log(std::max(p_[label], 1e-7f));

    // Softmax + cross-entropy has a simple combined gradient:
    // dz2 = p - onehot(label)
    for (int k = 0; k < out_; k++) {
        dz2_[k] = p_[k] - (k == label ? 1.0f : 0.0f);
    }

    // Back to the hidden layer: da1 = W2ᵀ·dz2.
    // Computed before W2 is updated below, since it needs the old weights.
    std::fill(da1_.begin(), da1_.end(), 0.0f);
    for (int k = 0; k < out_; k++) {
        for (int j = 0; j < hid_; j++) {
            da1_[j] += W2_[k * hid_ + j] * dz2_[k];
        }
    }

    // ReLU passes the gradient only where the unit was active.
    for (int j = 0; j < hid_; j++) {
        dz1_[j] = z1_[j] > 0.0f ? da1_[j] : 0.0f;
    }

    // SGD step for the output layer: dW2 = dz2 ⊗ a1, db2 = dz2
    for (int k = 0; k < out_; ++k) {
        for (int j = 0; j < hid_; ++j) {
            W2_[k * hid_ + j] -= lr * dz2_[k] * a1_[j];
        }
        b2_[k] -= lr * dz2_[k];
    }

    // SGD step for the hidden layer: dW1 = dz1 ⊗ x, db1 = dz1
    for (int j = 0; j < hid_; j++) {
        for (int i = 0; i < in_; i++) {
            W1_[j * in_ + i] -= lr * dz1_[j] * x_[i];
        }
        b1_[j] -= lr * dz1_[j];
    }

    return loss;
}

float Network::evaluate(const MnistDataset& ds, const std::vector<size_t>& idx) {
    int correct = 0;
    for (size_t n : idx) {
        const float* p = forward(ds.image(n));
        if (std::max_element(p, p + out_) - p == ds.label(n)) correct++;
    }
    return 100.0f * correct / idx.size();
}

float Network::train(const MnistDataset& ds, std::vector<size_t> train_idx,
                     const std::vector<size_t>& val_idx, int epochs, float lr) {
    float val_acc = 0.0f;

    for (int e = 0; e < epochs; ++e) {
        int correct = 0;
        float loss_sum = 0.0f;
        std::shuffle(train_idx.begin(), train_idx.end(), rng_);

        for (size_t n : train_idx) {
            const float* p = forward(ds.image(n));
            int y = ds.label(n);
            int pred = std::max_element(p, p + out_) - p;
            if (pred == y) correct++;
            loss_sum += backward(y, lr);
        }

        // Train accuracy is a running number: each prediction is made just
        // before that sample's update, so the weights change during the epoch.
        // Validation accuracy is measured on the weights at the end of the epoch.
        float train_acc = 100.0f * correct / train_idx.size();
        val_acc = evaluate(ds, val_idx);
        std::println("epoch {:2}/{} | loss {:.4f} | train {:.2f}% | val {:.2f}%",
                     e + 1, epochs, loss_sum / train_idx.size(), train_acc, val_acc);
    }

    return val_acc;
}
