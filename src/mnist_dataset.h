#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

// Loads an MNIST image/label file pair (IDX format) fully into memory.
// Throws std::runtime_error if a file is missing, malformed or truncated.
class MnistDataset {
public:
    static constexpr int kImageSize = 28 * 28;
    static constexpr int kNumClasses = 10;

    MnistDataset(const std::string& imagePath, const std::string& labelPath);
    size_t size() const;
    // Pointer to kImageSize grayscale pixels (0-255), row by row.
    const unsigned char* image(size_t index) const;
    uint8_t label(size_t index) const;

private:
    std::vector<unsigned char> images_;
    std::vector<uint8_t> labels_;
};