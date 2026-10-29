#include <algorithm>
#include <cstdio>
#include <exception>
#include <numeric>
#include <print>
#include <random>
#include <vector>
#include "mnist_dataset.h"
#include "network.h"

int main() {
	try {
		MnistDataset dataset("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
		std::println("loaded: {} samples", dataset.size());

		// Hold out 20% of the training set for validation. The fixed seed
		// gives the same split every run, so results are comparable.
		std::vector<size_t> idx(dataset.size());
		std::iota(idx.begin(), idx.end(), 0);
		std::shuffle(idx.begin(), idx.end(), std::mt19937(123));
		size_t n_train = idx.size() * 8 / 10;
		std::vector<size_t> train_idx(idx.begin(), idx.begin() + n_train);
		std::vector<size_t> val_idx(idx.begin() + n_train, idx.end());
		std::println("train: {}, val: {}", train_idx.size(), val_idx.size());

		Network net(MnistDataset::kImageSize, 64, MnistDataset::kNumClasses);
		net.train(dataset, train_idx, val_idx, 10, 0.01f);

		// The test set is only used once, after training is done.
		MnistDataset test_ds("data/t10k-images-idx3-ubyte",
		                     "data/t10k-labels-idx1-ubyte");
		std::vector<size_t> test_idx(test_ds.size());
		std::iota(test_idx.begin(), test_idx.end(), 0);
		std::println("test accuracy: {:.2f}%", net.evaluate(test_ds, test_idx));
	} catch (const std::exception& e) {
		std::println(stderr, "error: {}", e.what());
		return 1;
	}
	return 0;
}
