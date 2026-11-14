#pragma once

#include "common/Types.hpp"
#include <string>
#include <vector>
#include <filesystem>
#include <optional>

namespace gemfinder {

namespace fs = std::filesystem;

struct ProcessedImage {
    int original_width{0};
    int original_height{0};
    int original_channels{0};
    int target_width{224};
    int target_height{224};
    std::vector<float> rgb_planar_or_chw; // Float RGB values [0.0f, 1.0f]
    std::string preview_text;
    std::string document_prompt;
};

class ImagePreprocessor {
public:
    struct Config {
        int target_width{224};
        int target_height{224};
        bool normalize_zero_one{true}; // [0.0, 1.0] vs [-1.0, 1.0]
    };

    ImagePreprocessor();
    explicit ImagePreprocessor(Config config);

    // Load and preprocess image from file path
    [[nodiscard]] std::optional<ProcessedImage> processFile(const fs::path& path) const;

    // Load and preprocess image from memory buffer
    [[nodiscard]] std::optional<ProcessedImage> processMemory(const uint8_t* data, size_t length, const std::string& filename = "") const;

    // Fast header-only inspection without full decode
    [[nodiscard]] static bool getDimensions(const fs::path& path, int& width, int& height, int& channels);

private:
    Config config_;

    // High quality bilinear resizer
    static std::vector<float> resizeBilinear(
        const uint8_t* src,
        int src_w,
        int src_h,
        int src_channels,
        int dst_w,
        int dst_h,
        bool normalize_zero_one
    );
};

} // namespace gemfinder
