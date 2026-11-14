#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include "inference/InferenceEngine.hpp"
#include "inference/EngineFactory.hpp"
#include "inference/MockEngine.hpp"
#include "inference/LlamaEngine.hpp"
#include "inference/Normalization.hpp"
#include "common/Logger.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

void testNormalizationAndMrl() {
    gemfinder::Logger::info("--- Test 1: MRL Truncation & L2 Normalization ---");

    // Create 768-dimensional unnormalized vector
    std::vector<float> raw(768, 2.5f);
    auto mrl_vec = gemfinder::applyMrlAndNormalize(raw, gemfinder::kEmbeddingDim);

    assert(mrl_vec.size() == gemfinder::kEmbeddingDim);
    assert(mrl_vec.size() == 256);

    // Verify L2 norm is 1.0
    float norm_sq = 0.0f;
    for (float v : mrl_vec) {
        norm_sq += v * v;
    }
    float norm = std::sqrt(norm_sq);
    assert(std::abs(norm - 1.0f) < 1e-4f);

    gemfinder::Logger::info("MRL Vector Dim: " + std::to_string(mrl_vec.size()) +
                            ", L2 Norm: " + std::to_string(norm));
    gemfinder::Logger::info("Test 1 completato con successo!");
}

void testDeterministicMockEngine() {
    gemfinder::Logger::info("--- Test 2: Deterministic Mock Engine & Semantic Cosine Similarity ---");
    gemfinder::DeterministicMockEngine engine(256);

    assert(engine.isLoaded());
    assert(engine.dimension() == 256);

    std::string text_meteo_1 = "Previsioni meteo: forte temporale e pioggia a Milano";
    std::string text_meteo_2 = "Temporale estivo con forti precipitazioni di pioggia e grandine";
    std::string text_ricetta = "Ingredienti per preparare la pizza margherita con mozzarella e basilico";

    auto v_meteo_1 = engine.embedText(text_meteo_1);
    auto v_meteo_2 = engine.embedText(text_meteo_2);
    auto v_ricetta = engine.embedText(text_ricetta);

    float sim_similar = gemfinder::cosineSimilarity(v_meteo_1, v_meteo_2);
    float sim_dissimilar = gemfinder::cosineSimilarity(v_meteo_1, v_ricetta);

    gemfinder::Logger::info("Similarita' tra due testi sul meteo: " + std::to_string(sim_similar));
    gemfinder::Logger::info("Similarita' tra meteo e ricetta pizza: " + std::to_string(sim_dissimilar));

    assert(sim_similar > 0.30f);
    assert(sim_dissimilar < 0.25f);
    assert(sim_similar > sim_dissimilar);

    // Test image embedding
    std::vector<float> test_rgb(224 * 224 * 3, 0.5f);
    auto v_img = engine.embedImage(test_rgb, 224, 224);
    assert(v_img.size() == 256);

    // Test audio embedding
    std::vector<float> test_pcm(16000, 0.1f);
    auto v_aud = engine.embedAudio(test_pcm, 16000);
    assert(v_aud.size() == 256);

    gemfinder::Logger::info("Test 2 completato con successo!");
}

void testEngineFactory() {
    gemfinder::Logger::info("--- Test 3: Engine Factory & Graceful Fallback ---");

    // Factory with no model specified -> MockEngine
    auto engine1 = gemfinder::EngineFactory::createEngine(std::nullopt, 256);
    assert(engine1 != nullptr);
    assert(engine1->isLoaded());
    assert(engine1->dimension() == 256);
    assert(engine1->modelName() == "DeterministicMockEmbedder-v1");

    // Factory with non-existing path -> Graceful fallback
    auto engine2 = gemfinder::EngineFactory::createEngine(fs::path("/tmp/non_existing_model.gguf"), 256);
    assert(engine2 != nullptr);
    assert(engine2->isLoaded());
    assert(engine2->modelName() == "DeterministicMockEmbedder-v1");

    // Test LlamaEngine allocation without crash
    gemfinder::LlamaEngine llama_engine(256);
    assert(!llama_engine.isLoaded());
    assert(llama_engine.modelName() == "None");

    gemfinder::Logger::info("Test 3 completato con successo!");
}

int main() {
    gemfinder::Logger::info("=== Avvio Test Suite Inference Engine (Step 3) ===");
    testNormalizationAndMrl();
    testDeterministicMockEngine();
    testEngineFactory();
    gemfinder::Logger::info("=== TUTTI I TEST DELLO STEP 3 SUPERATI CON SUCCESSO! ===");
    return 0;
}
