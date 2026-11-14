#pragma once

#include "inference/InferenceEngine.hpp"
#include <memory>
#include <optional>
#include <filesystem>

namespace gemfinder {

namespace fs = std::filesystem;

class EngineFactory {
public:
    [[nodiscard]] static std::unique_ptr<InferenceEngine> createEngine(
        const std::optional<fs::path>& model_path = std::nullopt,
        size_t target_dimension = kEmbeddingDim
    );
};

} // namespace gemfinder
