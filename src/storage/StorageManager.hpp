#pragma once

#include "common/Types.hpp"
#include "storage/Database.hpp"
#include "storage/VectorStore.hpp"
#include <filesystem>
#include <optional>
#include <vector>
#include <span>
#include <memory>

namespace gemfinder {

namespace fs = std::filesystem;

class StorageManager {
public:
    StorageManager(const fs::path& db_path, const fs::path& vector_path, size_t vector_dim = kEmbeddingDim);
    ~StorageManager();

    // Initialization and Persistence
    bool initialize();
    bool save();

    // Indexing
    bool indexFile(FileRecord file, const std::vector<std::pair<ChunkRecord, Embedding>>& chunks_with_embeddings);
    bool removeFile(const fs::path& path);
    bool removeFileById(FileId file_id);
    size_t removeDirectory(const fs::path& dir_path);

    // Hybrid Search: vector + FTS5 BM25 with Reciprocal Rank Fusion (RRF)
    [[nodiscard]] std::vector<SearchResult> hybridSearch(
        const std::optional<std::string>& text_query,
        const std::optional<std::span<const float>>& vector_query,
        size_t limit = 20,
        double rrf_k = 60.0
    );

    // Diagnostics & Accessors
    [[nodiscard]] Database& database() noexcept { return db_; }
    [[nodiscard]] VectorStore& vectorStore() noexcept { return vectors_; }
    [[nodiscard]] size_t totalIndexedFiles();
    [[nodiscard]] size_t totalIndexedChunks() const noexcept { return vectors_.size(); }

private:
    Database db_;
    VectorStore vectors_;
    fs::path vector_path_;
};

} // namespace gemfinder
