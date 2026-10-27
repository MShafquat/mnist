#include <iostream>
#include "mnist_dataset.h"

int main() {
	MnistDataset dataset("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
	const unsigned char* p = dataset.image(0);
	for (int i = 0; i < 784; ++i) {
		std::cout << (p[i] > 128 ? '#' : '.');
		if ((i + 1) % 28 == 0) std::cout << '\n';
	}
	std::cout << "label=" << (int)dataset.label(0) << "\n";
}
