#pragma once

#include "inference/InferenceEngine.hpp"
#include <string>

namespace gemfinder {

class DeterministicMockEngine : public InferenceEngine {
public:
    explicit DeterministicMockEngine(size_t dimension = kEmbeddingDim);

    bool loadModel(const fs::path& model_path) override;
    [[nodiscard]] bool isLoaded() const noexcept override { return true; }
    [[nodiscard]] std::string modelName() const override { return "DeterministicMockEmbedder-v1"; }
    [[nodiscard]] size_t dimension() const noexcept override { return dimension_; }

    [[nodiscard]] Embedding embedText(std::string_view text) override;
    [[nodiscard]] Embedding embedImage(std::span<const float> rgb_pixels, int width, int height) override;
    [[nodiscard]] Embedding embedAudio(std::span<const float> pcm_samples, uint32_t sample_rate) override;

private:
    size_t dimension_{kEmbeddingDim};
};

} // namespace gemfinder
