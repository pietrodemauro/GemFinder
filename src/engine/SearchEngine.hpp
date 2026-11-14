#pragma once

#include "common/Types.hpp"
#include "storage/StorageManager.hpp"
#include "preprocessor/MediaPreprocessor.hpp"
#include "inference/InferenceEngine.hpp"
#include "engine/Crawler.hpp"

#include <memory>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

namespace gemfinder {

namespace fs = std::filesystem;

struct IndexStats {
    size_t scanned_files{0};
    size_t indexed_files{0};
    size_t skipped_files{0};
    size_t indexed_chunks{0};
    size_t error_files{0};
    double elapsed_seconds{0.0};
};

struct EngineStats {
    size_t total_files{0};
    size_t total_chunks{0};
    size_t vector_count{0};
    size_t embedding_dim{kEmbeddingDim};
    std::string inference_model;
    fs::path db_path;
    fs::path vector_path;
};

class SearchEngine {
public:
    SearchEngine(
        const fs::path& db_path,
        const fs::path& vector_path,
        std::unique_ptr<InferenceEngine> inference_engine
    );
    ~SearchEngine();

    // Initialization
    bool initialize();
    bool save();

    // Directory and file indexing
    IndexStats indexDirectory(
        const fs::path& root_path,
        bool force_reindex = false,
        const std::function<void(const std::string& current_file, size_t current, size_t total)>& on_progress = nullptr
    );

    bool indexSingleFile(const fs::path& path);
    bool removeFile(const fs::path& path);

    // Queries
    [[nodiscard]] std::vector<SearchResult> search(
        std::string_view query_text,
        size_t limit = 20,
        double rrf_k = 60.0
    );

    [[nodiscard]] std::vector<SearchResult> searchByImage(
        const fs::path& image_path,
        size_t limit = 20
    );

    [[nodiscard]] std::vector<SearchResult> searchByAudio(
        const fs::path& audio_path,
        size_t limit = 20
    );

    // Statistics & diagnostics
    [[nodiscard]] EngineStats getStats();

    [[nodiscard]] InferenceEngine& inference() noexcept { return *inference_; }
    [[nodiscard]] StorageManager& storage() noexcept { return *storage_; }
    [[nodiscard]] MediaPreprocessor& preprocessor() noexcept { return preprocessor_; }

private:
    std::unique_ptr<StorageManager> storage_;
    std::unique_ptr<InferenceEngine> inference_;
    MediaPreprocessor preprocessor_;
    Crawler crawler_;
    fs::path db_path_;
    fs::path vector_path_;
};

} // namespace gemfinder
