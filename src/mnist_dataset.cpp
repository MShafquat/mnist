#include "mnist_dataset.h"
#include <bit>
#include <fstream>
#include <stdexcept>

namespace {

// IDX headers store integers big-endian.
uint32_t read_be32(std::ifstream& in) {
    uint32_t x = 0;
    in.read(reinterpret_cast<char*>(&x), 4);
    if constexpr (std::endian::native == std::endian::little) x = std::byteswap(x);
    return x;
}

}  // namespace

MnistDataset::MnistDataset(const std::string& imagePath, const std::string& labelPath) {
    std::ifstream img(imagePath, std::ios::binary);
    if (!img) throw std::runtime_error("Cannot open " + imagePath);
    uint32_t magic = read_be32(img);
    uint32_t count = read_be32(img);
    uint32_t rows = read_be32(img);
    uint32_t cols = read_be32(img);

    if (magic != 2051 || rows != 28 || cols != 28) {
        throw std::runtime_error("bad image file loaded");
    }

    images_.resize(static_cast<size_t>(count) * kImageSize);
    img.read(reinterpret_cast<char*>(images_.data()), images_.size());
    if (!img) throw std::runtime_error("image read truncated!");

    std::ifstream lbl(labelPath, std::ios::binary);
    if (!lbl) throw std::runtime_error("Cannot open " + labelPath);

    uint32_t lmagic = read_be32(lbl);
    uint32_t lcount = read_be32(lbl);

    if (lmagic != 2049 || lcount != count) {
        throw std::runtime_error("Bad or mismatched label file header!");
    }

    labels_.resize(lcount);
    lbl.read(reinterpret_cast<char*>(labels_.data()), labels_.size());
    if (!lbl) throw std::runtime_error("label read truncated!");
}

size_t MnistDataset::size() const { return labels_.size(); }

std::span<const uint8_t, MnistDataset::kImageSize> MnistDataset::image(size_t index) const {
    return std::span<const uint8_t, kImageSize>(images_.data() + index * kImageSize, kImageSize);
}

uint8_t MnistDataset::label(size_t index) const {
    return labels_[index];
}
