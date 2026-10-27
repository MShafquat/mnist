#include "mnist_dataset.h"
#include <arpa/inet.h>
#include <fstream>

MnistDataset::MnistDataset(const std::string& imagePath, const std::string& labelPath) {
    std::ifstream img(imagePath, std::ios::binary);
    if (!img) throw std::runtime_error("Cannot open " + imagePath);
    uint32_t magic, count, rows, cols;
    img.read(reinterpret_cast<char*>(&magic), 4); magic = ntohl(magic);
    img.read(reinterpret_cast<char*>(&count), 4); count = ntohl(count);
    img.read(reinterpret_cast<char*>(&rows), 4); rows = ntohl(rows);
    img.read(reinterpret_cast<char*>(&cols), 4); cols = ntohl(cols);

    if (magic != 2051 || rows != 28 || cols != 28) {
        throw std::runtime_error("bad image file loaded");
    }

    images_.resize(static_cast<size_t>(count) * 784);
    img.read(reinterpret_cast<char*>(images_.data()), images_.size());
    if (!img) throw std::runtime_error("image read truncated!");

    std::ifstream lbl(labelPath, std::ios::binary);
    if (!lbl) throw std::runtime_error("Cannot open " + labelPath);

    uint32_t lmagic, lcount;
    lbl.read(reinterpret_cast<char*>(&lmagic), 4); lmagic = ntohl(lmagic);
    lbl.read(reinterpret_cast<char*>(&lcount), 4); lcount = ntohl(lcount);

    if (lmagic != 2049 || lcount != count) {
        throw std::runtime_error("Bad or mismatched label file header!");
    }

    labels_.resize(lcount);
    lbl.read(reinterpret_cast<char*>(labels_.data()), labels_.size());
    if (!lbl) throw std::runtime_error("label read truncated!");
}

size_t MnistDataset::size() const { return labels_.size(); }

const unsigned char* MnistDataset::image(size_t index) const {
    return images_.data() + index * 784;
}

uint8_t MnistDataset::label(size_t index) const {
    return labels_[index];
}