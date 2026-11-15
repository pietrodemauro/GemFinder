#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include "inference/Normalization.hpp"
#include "inference/MockEngine.hpp"
#include "inference/LlamaEngine.hpp"
#include "inference/EngineFactory.hpp"
#include "preprocessor/ImagePreprocessor.hpp"
#include "common/Logger.hpp"
#include <cmath>
#include <vector>
#include <iostream>
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
    gemfinder::Logger::info("Test 1 completed successfully!");
}

void testDeterministicMockEngine() {
    gemfinder::Logger::info("--- Test 2: Deterministic Mock Engine & Semantic Cosine Similarity ---");
    gemfinder::DeterministicMockEngine engine(256);

    assert(engine.isLoaded());
    assert(engine.dimension() == 256);

    std::string text_weather_1 = "Weather forecast: heavy thunderstorm and torrential rainfall in the city";
    std::string text_weather_2 = "Summer weather forecast with thunderstorm and torrential rainfall";
    std::string text_recipe = "Ingredients for preparing classic pizza margherita with tomato and mozzarella cheese";

    auto v_weather_1 = engine.embedText(text_weather_1);
    auto v_weather_2 = engine.embedText(text_weather_2);
    auto v_recipe = engine.embedText(text_recipe);

    float sim_similar = gemfinder::cosineSimilarity(v_weather_1, v_weather_2);
    float sim_dissimilar = gemfinder::cosineSimilarity(v_weather_1, v_recipe);

    gemfinder::Logger::info("Similarity between two weather texts: " + std::to_string(sim_similar));
    gemfinder::Logger::info("Similarity between weather and recipe: " + std::to_string(sim_dissimilar));

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

    gemfinder::Logger::info("Test 2 completed successfully!");
}

void testEngineFactory() {
    gemfinder::Logger::info("--- Test 3: Engine Factory & Graceful Fallback ---");

    // Factory with no model specified -> Loads default GGUF if present, or MockEngine
    auto engine1 = gemfinder::EngineFactory::createEngine(std::nullopt, 256);
    assert(engine1 != nullptr);
    assert(engine1->isLoaded());
    assert(engine1->dimension() == 256);

    // Factory with non-existing path -> Graceful fallback
    auto engine2 = gemfinder::EngineFactory::createEngine(fs::path("/tmp/non_existing_model.gguf"), 256);
    assert(engine2 != nullptr);
    assert(engine2->isLoaded());
    assert(engine2->modelName() == "DeterministicMockEmbedder-v1");

    // Test LlamaEngine allocation without crash
    gemfinder::LlamaEngine llama_engine(256);
    assert(!llama_engine.isLoaded());
    assert(llama_engine.modelName() == "None");

    gemfinder::Logger::info("Test 3 completed successfully!");
}

void testRealMultimodalImageEmbedding() {
    if (!fs::exists("models/embeddinggemma-2.gguf") || !fs::exists("models/mmproj.gguf")) {
        gemfinder::Logger::info("Skipping real multimodal test (models not found)");
        return;
    }
    gemfinder::Logger::info("--- Test 4: Real Multimodal Image Embedding with EmbeddingGemma 2 & mmproj ---");
    auto engine = gemfinder::EngineFactory::createEngine(fs::path("models/embeddinggemma-2.gguf"), 256);
    assert(engine != nullptr);
    assert(engine->isLoaded());

    // Process test images if they exist
    if (!fs::exists("/tmp/test_red.png") || !fs::exists("/tmp/test_red_similar.png") || !fs::exists("/tmp/test_blue.png")) {
        gemfinder::Logger::info("Skipping image similarity checks (test images not present in /tmp)");
        return;
    }

    gemfinder::ImagePreprocessor preprocessor;
    auto img1 = preprocessor.processFile("/tmp/test_red.png");
    auto img2 = preprocessor.processFile("/tmp/test_red_similar.png");
    auto img3 = preprocessor.processFile("/tmp/test_blue.png");
    assert(img1.has_value() && img2.has_value() && img3.has_value());

    auto emb1 = engine->embedImage(img1->rgb_planar_or_chw, img1->target_width, img1->target_height);
    auto emb2 = engine->embedImage(img2->rgb_planar_or_chw, img2->target_width, img2->target_height);
    auto emb3 = engine->embedImage(img3->rgb_planar_or_chw, img3->target_width, img3->target_height);

    float sim_similar = gemfinder::cosineSimilarity(emb1, emb2);
    float sim_different = gemfinder::cosineSimilarity(emb1, emb3);

    gemfinder::Logger::info("Similarity between red and similar red image: " + std::to_string(sim_similar));
    gemfinder::Logger::info("Similarity between red and blue image:        " + std::to_string(sim_different));

    assert(sim_similar > sim_different);
    gemfinder::Logger::info("Test 4 completed successfully!");
}

int main() {
    gemfinder::Logger::info("=== Starting Inference Engine Test Suite (Step 3) ===");
    testNormalizationAndMrl();
    testDeterministicMockEngine();
    testEngineFactory();
    testRealMultimodalImageEmbedding();
    gemfinder::Logger::info("=== ALL STEP 3 TESTS PASSED SUCCESSFULLY! ===");
    return 0;
}
