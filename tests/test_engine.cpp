#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include "engine/Crawler.hpp"
#include "engine/SearchEngine.hpp"
#include "inference/MockEngine.hpp"
#include "common/Logger.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void testCrawler() {
    gemfinder::Logger::info("--- Test 1: Crawler Directory Scanning & Filters ---");
    fs::path temp_dir = "/tmp/test_gemfinder_crawler";
    fs::remove_all(temp_dir);
    fs::create_directories(temp_dir / "src");
    fs::create_directories(temp_dir / ".git");
    fs::create_directories(temp_dir / "node_modules");

    // Create files
    std::ofstream(temp_dir / "src" / "main.cpp") << "int main() {}";
    std::ofstream(temp_dir / "src" / "header.hpp") << "#pragma once";
    std::ofstream(temp_dir / ".git" / "config") << "[core]";
    std::ofstream(temp_dir / "node_modules" / "package.json") << "{}";
    std::ofstream(temp_dir / ".hidden_file.txt") << "secret";

    gemfinder::Crawler crawler;
    auto found_files = crawler.crawl(temp_dir);

    assert(found_files.size() == 2);
    gemfinder::Logger::info("Crawler found " + std::to_string(found_files.size()) + " files (excluded .git and node_modules)");

    fs::remove_all(temp_dir);
    gemfinder::Logger::info("Test 1 completed successfully!");
}

void testSearchEngineEndToEnd() {
    gemfinder::Logger::info("--- Test 2: SearchEngine End-to-End Hybrid Search ---");
    fs::path test_dir = "/tmp/test_gemfinder_engine_docs";
    fs::path test_db = "/tmp/test_gemfinder_engine.db";
    fs::path test_vec = "/tmp/test_gemfinder_engine.usearch";

    fs::remove_all(test_dir);
    fs::remove(test_db);
    fs::remove(test_vec);
    fs::create_directories(test_dir);

    // Create 3 diverse files
    std::ofstream(test_dir / "weather.txt")
        << "Weather forecast: heavy thunderstorm with torrential rain and flooding across the city.";

    std::ofstream(test_dir / "vectors.cpp")
        << "HNSW graph algorithm for ultra-fast approximate nearest neighbor vector search in C++20.";

    std::ofstream(test_dir / "recipe.md")
        << "Classic Neapolitan pizza margherita recipe with flour, San Marzano tomatoes, and mozzarella.";

    auto mock_inf = std::make_unique<gemfinder::DeterministicMockEngine>(gemfinder::kEmbeddingDim);
    gemfinder::SearchEngine engine(test_db, test_vec, std::move(mock_inf));
    assert(engine.initialize());

    // 1. Index Directory
    auto stats = engine.indexDirectory(test_dir);
    assert(stats.indexed_files == 3);
    assert(stats.scanned_files == 3);
    assert(stats.skipped_files == 0);
    gemfinder::Logger::info("Indexed " + std::to_string(stats.indexed_files) + " files in " +
                            std::to_string(stats.elapsed_seconds) + "s");

    // 2. Hybrid Search for Weather
    auto results_weather = engine.search("thunderstorm rain", 5);
    assert(!results_weather.empty());
    assert(results_weather[0].filename == "weather.txt");
    gemfinder::Logger::info("Top match for 'thunderstorm rain': " + results_weather[0].filename +
                            " (Score: " + std::to_string(results_weather[0].score) + ")");

    // 3. Hybrid Search for Vectors
    auto results_vec = engine.search("HNSW vectors", 5);
    assert(!results_vec.empty());
    assert(results_vec[0].filename == "vectors.cpp");
    gemfinder::Logger::info("Top match for 'HNSW vectors': " + results_vec[0].filename +
                            " (Score: " + std::to_string(results_vec[0].score) + ")");

    // 4. Test Incremental Indexing (skipping unmodified files)
    auto stats_incremental = engine.indexDirectory(test_dir);
    assert(stats_incremental.skipped_files == 3);
    assert(stats_incremental.indexed_files == 0);
    gemfinder::Logger::info("Incremental scan correctly skipped all " +
                            std::to_string(stats_incremental.skipped_files) + " unmodified files!");

    // 5. Test File Deletion
    assert(engine.removeFile(test_dir / "recipe.md"));
    auto stats_after_del = engine.getStats();
    assert(stats_after_del.total_files == 2);
    auto results_pizza = engine.search("pizza margherita", 5);
    for (const auto& r : results_pizza) {
        assert(r.filename != "recipe.md");
    }
    gemfinder::Logger::info("File deletion and index update verified!");

    fs::remove_all(test_dir);
    fs::remove(test_db);
    fs::remove(test_vec);
    gemfinder::Logger::info("Test 2 completed successfully!");
}

int main() {
    gemfinder::Logger::info("=== Starting Search Engine Test Suite (Step 4) ===");
    testCrawler();
    testSearchEngineEndToEnd();
    gemfinder::Logger::info("=== ALL STEP 4 TESTS PASSED SUCCESSFULLY! ===");
    return 0;
}
