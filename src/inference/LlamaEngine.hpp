#pragma once

#include "inference/InferenceEngine.hpp"
#include <memory>
#include <mutex>

struct llama_model;
struct llama_context;

namespace gemfinder {

class LlamaEngine : public InferenceEngine {
public:
    explicit LlamaEngine(size_t target_dimension = kEmbeddingDim, int n_threads = 0);
    ~LlamaEngine() override;

    // Non-copyable, movable
    LlamaEngine(const LlamaEngine&) = delete;
    LlamaEngine& operator=(const LlamaEngine&) = delete;
    LlamaEngine(LlamaEngine&&) noexcept;
    LlamaEngine& operator=(LlamaEngine&&) noexcept;

    bool loadModel(const fs::path& model_path) override;
    [[nodiscard]] bool isLoaded() const noexcept override;
    [[nodiscard]] std::string modelName() const override;
    [[nodiscard]] size_t dimension() const noexcept override { return target_dim_; }

    [[nodiscard]] Embedding embedText(std::string_view text) override;
    [[nodiscard]] Embedding embedImage(std::span<const float> rgb_pixels, int width, int height) override;
    [[nodiscard]] Embedding embedAudio(std::span<const float> pcm_samples, uint32_t sample_rate) override;

    [[nodiscard]] size_t rawModelDimension() const noexcept;

private:
    llama_model* model_{nullptr};
    llama_context* ctx_{nullptr};
    size_t target_dim_{kEmbeddingDim};
    int n_threads_{0};
    std::string model_name_;
    mutable std::mutex mutex_;

    void unload();
};

} // namespace gemfinder
