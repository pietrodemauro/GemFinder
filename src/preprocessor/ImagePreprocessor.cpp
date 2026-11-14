#include "preprocessor/ImagePreprocessor.hpp"
#include "common/Logger.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO_NOT_REALLY
#include <stb/stb_image.h>

#include <cmath>
#include <algorithm>

namespace gemfinder {

ImagePreprocessor::ImagePreprocessor() : config_{} {}
ImagePreprocessor::ImagePreprocessor(Config config) : config_(config) {}

bool ImagePreprocessor::getDimensions(const fs::path& path, int& width, int& height, int& channels) {
    return stbi_info(path.c_str(), &width, &height, &channels) != 0;
}

std::vector<float> ImagePreprocessor::resizeBilinear(
    const uint8_t* src,
    int src_w,
    int src_h,
    int src_channels,
    int dst_w,
    int dst_h,
    bool normalize_zero_one
) {
    std::vector<float> dst(static_cast<size_t>(dst_w * dst_h * 3));

    const float x_scale = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float y_scale = static_cast<float>(src_h) / static_cast<float>(dst_h);

    for (int dy = 0; dy < dst_h; ++dy) {
        float sy = (static_cast<float>(dy) + 0.5f) * y_scale - 0.5f;
        int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, src_h - 1);
        int y1 = std::clamp(y0 + 1, 0, src_h - 1);
        float wy = sy - static_cast<float>(y0);

        for (int dx = 0; dx < dst_w; ++dx) {
            float sx = (static_cast<float>(dx) + 0.5f) * x_scale - 0.5f;
            int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, src_w - 1);
            int x1 = std::clamp(x0 + 1, 0, src_w - 1);
            float wx = sx - static_cast<float>(x0);

            for (int c = 0; c < 3; ++c) {
                // If grayscale, replicate across RGB; otherwise pick channel
                int ch = (src_channels >= 3) ? c : 0;

                float p00 = static_cast<float>(src[(y0 * src_w + x0) * src_channels + ch]);
                float p10 = static_cast<float>(src[(y0 * src_w + x1) * src_channels + ch]);
                float p01 = static_cast<float>(src[(y1 * src_w + x0) * src_channels + ch]);
                float p11 = static_cast<float>(src[(y1 * src_w + x1) * src_channels + ch]);

                float top = p00 * (1.0f - wx) + p10 * wx;
                float bottom = p01 * (1.0f - wx) + p11 * wx;
                float val = top * (1.0f - wy) + bottom * wy;

                // Normalize to [0.0, 1.0] or [-1.0, 1.0]
                float normalized = val / 255.0f;
                if (!normalize_zero_one) {
                    normalized = normalized * 2.0f - 1.0f;
                }

                dst[static_cast<size_t>((dy * dst_w + dx) * 3 + c)] = normalized;
            }
        }
    }

    return dst;
}

std::optional<ProcessedImage> ImagePreprocessor::processFile(const fs::path& path) const {
    int w = 0, h = 0, channels = 0;
    // Request 3 channels (RGB)
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!data) {
        Logger::error("stb_image failed to load: " + path.string() + " (" + stbi_failure_reason() + ")");
        return std::nullopt;
    }

    ProcessedImage img;
    img.original_width = w;
    img.original_height = h;
    img.original_channels = channels;
    img.target_width = config_.target_width;
    img.target_height = config_.target_height;

    img.rgb_planar_or_chw = resizeBilinear(
        data, w, h, 3, config_.target_width, config_.target_height, config_.normalize_zero_one
    );

    stbi_image_free(data);

    img.preview_text = "Image [" + std::to_string(w) + "x" + std::to_string(h) + "] " + path.filename().string();
    img.document_prompt = "title: " + path.filename().string() + " | <|image|>";

    return img;
}

std::optional<ProcessedImage> ImagePreprocessor::processMemory(const uint8_t* data, size_t length, const std::string& filename) const {
    if (!data || length == 0) return std::nullopt;

    int w = 0, h = 0, channels = 0;
    uint8_t* decoded = stbi_load_from_memory(data, static_cast<int>(length), &w, &h, &channels, 3);
    if (!decoded) {
        return std::nullopt;
    }

    ProcessedImage img;
    img.original_width = w;
    img.original_height = h;
    img.original_channels = channels;
    img.target_width = config_.target_width;
    img.target_height = config_.target_height;

    img.rgb_planar_or_chw = resizeBilinear(
        decoded, w, h, 3, config_.target_width, config_.target_height, config_.normalize_zero_one
    );

    stbi_image_free(decoded);

    img.preview_text = "Image [" + std::to_string(w) + "x" + std::to_string(h) + "] " + filename;
    img.document_prompt = "title: " + filename + " | <|image|>";

    return img;
}

} // namespace gemfinder
