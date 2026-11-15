#include "inference/EngineFactory.hpp"
#include "inference/LlamaEngine.hpp"
#include "inference/MockEngine.hpp"
#include "common/Logger.hpp"

#include <cstdlib>

namespace gemfinder {

std::optional<fs::path> EngineFactory::defaultModelPath() {
    std::vector<fs::path> candidates = {
        fs::current_path() / "models" / "embeddinggemma-2.gguf",
        fs::current_path() / "models" / "embeddinggemma-2-BF16.gguf",
    };
    const char* home = std::getenv("HOME");
    if (home) {
        candidates.push_back(fs::path(home) / ".local/share/gemfinder/models/embeddinggemma-2.gguf");
        candidates.push_back(fs::path(home) / ".local/share/gemfinder/models/embeddinggemma-2-BF16.gguf");
    }

    for (const auto& c : candidates) {
        if (!c.empty() && fs::exists(c)) {
            return c;
        }
    }

    if (fs::exists("models") && fs::is_directory("models")) {
        for (const auto& entry : fs::directory_iterator("models")) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                std::string fname = entry.path().filename().string();
                if (fname.find("mmproj") == std::string::npos) {
                    return entry.path();
                }
            }
        }
    }

    return std::nullopt;
}

std::unique_ptr<InferenceEngine> EngineFactory::createEngine(
    const std::optional<fs::path>& model_path,
    size_t target_dimension,
    int n_threads,
    bool use_gpu
) {
    std::optional<fs::path> chosen_path = model_path;
    if (!chosen_path.has_value() || chosen_path->empty()) {
        chosen_path = defaultModelPath();
        if (chosen_path.has_value()) {
            Logger::info("EngineFactory: Found default GGUF model: " + chosen_path->string());
        }
    }

    if (chosen_path.has_value() && !chosen_path->empty()) {
        if (fs::exists(*chosen_path)) {
            auto llama = std::make_unique<LlamaEngine>(target_dimension, n_threads, use_gpu);
            if (llama->loadModel(*chosen_path)) {
                return llama;
            }
            Logger::warn("Unable to load GGUF model: " + chosen_path->string() + ". Falling back to DeterministicMockEngine.");
        } else {
            Logger::warn("Model path not found: " + chosen_path->string() + ". Falling back to DeterministicMockEngine.");
        }
    } else {
        Logger::info("No GGUF model specified or found in default paths. Using 256-dim DeterministicMockEngine.");
    }

    return std::make_unique<DeterministicMockEngine>(target_dimension);
}

} // namespace gemfinder
