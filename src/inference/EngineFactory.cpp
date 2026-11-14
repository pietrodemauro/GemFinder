#include "inference/EngineFactory.hpp"
#include "inference/LlamaEngine.hpp"
#include "inference/MockEngine.hpp"
#include "common/Logger.hpp"

namespace gemfinder {

std::unique_ptr<InferenceEngine> EngineFactory::createEngine(
    const std::optional<fs::path>& model_path,
    size_t target_dimension
) {
    if (model_path.has_value() && !model_path->empty()) {
        if (fs::exists(*model_path)) {
            auto llama = std::make_unique<LlamaEngine>(target_dimension);
            if (llama->loadModel(*model_path)) {
                return llama;
            }
            Logger::warn("Impossibile caricare il modello GGUF: " + model_path->string() + ". Fallback al DeterministicMockEngine.");
        } else {
            Logger::warn("Percorso modello non trovato: " + model_path->string() + ". Fallback al DeterministicMockEngine.");
        }
    } else {
        Logger::info("Nessun modello GGUF specificato (--model). Uso del DeterministicMockEngine a 256 dimensioni.");
    }

    return std::make_unique<DeterministicMockEngine>(target_dimension);
}

} // namespace gemfinder
