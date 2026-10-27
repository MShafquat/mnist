#include <cstdint>
#include <string>
#include <vector>

class MnistDataset {
public:
    MnistDataset(const std::string& imagePath, const std::string& labelPath);
    size_t size() const;
    const unsigned char* image(size_t index) const;
    uint8_t label(size_t index) const;

private:
    std::vector<unsigned char> images_;
    std::vector<uint8_t> labels_;
};