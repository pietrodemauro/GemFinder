#include "storage/StorageManager.hpp"
#include "common/Logger.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;

// Helper to create a normalized 256-dimensional vector with a peak at specific dimension
gemfinder::Embedding makeSyntheticVector(size_t primary_dim, float weight = 1.0f) {
    gemfinder::Embedding vec(gemfinder::kEmbeddingDim, 0.01f);
    if (primary_dim < gemfinder::kEmbeddingDim) {
        vec[primary_dim] = weight;
    }
    // L2 Normalize
    float norm_sq = 0.0f;
    for (float v : vec) norm_sq += v * v;
    float norm = std::sqrt(norm_sq);
    for (float& v : vec) v /= norm;
    return vec;
}

void testDatabaseAndFTS() {
    gemfinder::Logger::info("--- Test 1: Database & FTS5 ---");
    fs::path test_db = "/tmp/test_gemfinder.db";
    fs::remove(test_db);

    gemfinder::Database db(test_db);
    assert(db.isOpen());
    assert(db.initializeSchema());

    // 1. Insert file
    gemfinder::FileRecord file1;
    file1.filepath = "/tmp/appunti_meteo.txt";
    file1.filename = "appunti_meteo.txt";
    file1.extension = ".txt";
    file1.file_size = 1024;
    file1.mtime = 1700000000;
    file1.xxhash64 = 0x12345678;
    file1.modality = gemfinder::Modality::Text;

    auto file_id_opt = db.upsertFile(file1);
    assert(file_id_opt.has_value());
    gemfinder::FileId fid = *file_id_opt;

    // 2. Insert chunks
    gemfinder::ChunkRecord chunk1;
    chunk1.file_id = fid;
    chunk1.chunk_type = "text_paragraph";
    chunk1.start_offset = 0;
    chunk1.end_offset = 150;
    chunk1.preview_text = "Oggi a Milano piove forte, previsto temporale con grandine nel pomeriggio.";
    auto cid1 = db.insertChunk(chunk1);
    assert(cid1.has_value());

    gemfinder::ChunkRecord chunk2;
    chunk2.file_id = fid;
    chunk2.chunk_type = "text_paragraph";
    chunk2.start_offset = 151;
    chunk2.end_offset = 300;
    chunk2.preview_text = "Domani splenderà il sole e le temperature aumenteranno.";
    auto cid2 = db.insertChunk(chunk2);
    assert(cid2.has_value());

    // 3. Search FTS
    auto fts_results = db.searchFts("temporale");
    assert(!fts_results.empty());
    assert(fts_results[0].chunk_id == *cid1);
    gemfinder::Logger::info("FTS5 Match: Found chunk_id=" + std::to_string(fts_results[0].chunk_id) +
                            " Snippet: " + fts_results[0].snippet);

    fs::remove(test_db);
    gemfinder::Logger::info("Test 1 completato con successo!");
}

void testVectorStore() {
    gemfinder::Logger::info("--- Test 2: VectorStore (USearch) ---");
    fs::path index_path = "/tmp/test_vectors.usearch";
    fs::remove(index_path);

    gemfinder::VectorStore vs(gemfinder::kEmbeddingDim);
    assert(vs.size() == 0);

    auto v1 = makeSyntheticVector(10, 5.0f);
    auto v2 = makeSyntheticVector(20, 5.0f);
    auto v3 = makeSyntheticVector(10, 4.0f); // Close to v1

    assert(vs.add(101, v1));
    assert(vs.add(102, v2));
    assert(vs.add(103, v3));
    assert(vs.size() == 3);

    // Search near v1
    auto query = makeSyntheticVector(10, 5.0f);
    auto matches = vs.search(query, 2);
    assert(matches.size() == 2);
    assert(matches[0].chunk_id == 101); // Identical
    assert(matches[0].similarity > 0.99f);
    assert(matches[1].chunk_id == 103); // Close

    // Persistence & reload
    assert(vs.save(index_path));

    gemfinder::VectorStore vs_loaded(gemfinder::kEmbeddingDim);
    assert(vs_loaded.load(index_path));
    assert(vs_loaded.size() == 3);

    gemfinder::VectorStore vs_view(gemfinder::kEmbeddingDim);
    assert(vs_view.view(index_path));
    assert(vs_view.size() == 3);

    fs::remove(index_path);
    gemfinder::Logger::info("Test 2 completato con successo!");
}

void testHybridSearch() {
    gemfinder::Logger::info("--- Test 3: Hybrid Search with RRF Fusion ---");
    fs::path test_db = "/tmp/test_hybrid.db";
    fs::path test_vec = "/tmp/test_hybrid.usearch";
    fs::remove(test_db);
    fs::remove(test_vec);

    gemfinder::StorageManager storage(test_db, test_vec, gemfinder::kEmbeddingDim);
    assert(storage.initialize());

    // File A: Document on AI and machine learning
    gemfinder::FileRecord doc_a;
    doc_a.filepath = "/docs/ai_architecture.md";
    doc_a.filename = "ai_architecture.md";
    doc_a.extension = ".md";
    gemfinder::ChunkRecord chunk_a;
    chunk_a.chunk_type = "text_paragraph";
    chunk_a.preview_text = "EmbeddingGemma 2 è un modello per la ricerca semantica multimodale ad alta velocità.";
    auto vec_a = makeSyntheticVector(5, 5.0f);

    assert(storage.indexFile(doc_a, {{chunk_a, vec_a}}));

    // File B: Document on Italian cuisine
    gemfinder::FileRecord doc_b;
    doc_b.filepath = "/docs/ricetta_pizza.txt";
    doc_b.filename = "ricetta_pizza.txt";
    doc_b.extension = ".txt";
    gemfinder::ChunkRecord chunk_b;
    chunk_b.chunk_type = "text_paragraph";
    chunk_b.preview_text = "Impasto per pizza napoletana a lunga lievitazione e cottura ad alta temperatura.";
    auto vec_b = makeSyntheticVector(50, 5.0f);

    assert(storage.indexFile(doc_b, {{chunk_b, vec_b}}));

    // File C: Document on climate
    gemfinder::FileRecord doc_c;
    doc_c.filepath = "/docs/clima_meteo.txt";
    doc_c.filename = "clima_meteo.txt";
    doc_c.extension = ".txt";
    gemfinder::ChunkRecord chunk_c;
    chunk_c.chunk_type = "text_paragraph";
    chunk_c.preview_text = "Temporale estivo con forti precipitazioni e grandine.";
    auto vec_c = makeSyntheticVector(100, 5.0f);

    assert(storage.indexFile(doc_c, {{chunk_c, vec_c}}));
    assert(storage.totalIndexedFiles() == 3);

    // Hybrid query: vector close to AI (dim 5) + lexical search "modello"
    auto query_vec = makeSyntheticVector(5, 4.0f);
    auto results = storage.hybridSearch("modello", query_vec, 10);

    assert(!results.empty());
    // Document A should be top-1 with highest RRF score because it matched BOTH vector and text!
    assert(results[0].filename == "ai_architecture.md");
    gemfinder::Logger::info("Top 1 Match: " + results[0].filename +
                            " RRF Score: " + std::to_string(results[0].score) +
                            " (Vector Rank: " + std::to_string(results[0].rank_vector) +
                            ", Lexical Rank: " + std::to_string(results[0].rank_lexical) + ")");

    fs::remove(test_db);
    fs::remove(test_vec);
    gemfinder::Logger::info("Test 3 completato con successo!");
}

int main() {
    gemfinder::Logger::info("=== Avvio Test Suite Storage & Hybrid Search ===");
    testDatabaseAndFTS();
    testVectorStore();
    testHybridSearch();
    gemfinder::Logger::info("=== TUTTI I TEST DELLO STEP 1 SUPERATI CON SUCCESSO! ===");
    return 0;
}
