#include <iostream>
#include <fstream>
#include <array>
#include <cstddef>
#include <iomanip>
#include <arpa/inet.h>
#include <vector>

struct Header {
	uint32_t magic;
	uint32_t count;
	uint32_t rows;
	uint32_t cols;
};

int main() {
	std::ifstream input("../data/train-images-idx3-ubyte", std::ios::binary);
	std::ifstream labels("../data/train-labels-idx1-ubyte", std::ios::binary);

	if (!input && !labels) {
		std::cerr << "Failed to open the file\n";
		return 1;
	}


	Header h;
	input.read(reinterpret_cast<char*>(&h), sizeof(h));
	h.magic = ntohl(h.magic);
	h.count = ntohl(h.count);
	h.rows = ntohl(h.rows);
	h.cols = ntohl(h.cols);

	std::cout << "magic=" << h.magic
          << " count=" << h.count
          << " rows="  << h.rows
          << " cols="  << h.cols << "\n";

	std::vector<unsigned char> pixels(784);
	input.read(reinterpret_cast<char*>(pixels.data()), pixels.size());

	for (int i = 0; i < 784; ++i) {
		std::cout << (pixels[i] > 128 ? '#' : '.');
		if ((i + 1) % 28 == 0) std::cout << '\n';
	}

	uint32_t lmagic, lcount;
	labels.read(reinterpret_cast<char*>(&lmagic), sizeof(lmagic));
	labels.read(reinterpret_cast<char*>(&lcount), sizeof(lcount));

	lmagic = ntohl(lmagic);
	lcount = ntohl(lcount);
	std::cout << "Magic=" << lmagic << " Count=" << lcount << "\n";

	unsigned char label;
	labels.read(reinterpret_cast<char*>(&label), 1);
	std::cout << "Label: " << (int) label << "\n";

	return 0;
}
