#pragma once

#include "common/Types.hpp"
#include <string_view>
#include <vector>
#include <span>
#include <filesystem>
#include <string>

namespace gemfinder {

namespace fs = std::filesystem;

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    // Model lifecycle
    virtual bool loadModel(const fs::path& model_path) = 0;
    [[nodiscard]] virtual bool isLoaded() const noexcept = 0;
    [[nodiscard]] virtual std::string modelName() const = 0;
    [[nodiscard]] virtual size_t dimension() const noexcept = 0;

    // Multimodal Embedding generation (always returns 256d L2-normalized vector)
    [[nodiscard]] virtual Embedding embedText(std::string_view text) = 0;
    [[nodiscard]] virtual Embedding embedImage(std::span<const float> rgb_pixels, int width, int height) = 0;
    [[nodiscard]] virtual Embedding embedAudio(std::span<const float> pcm_samples, uint32_t sample_rate) = 0;
};

} // namespace gemfinder
