#pragma once

#include "common/Types.hpp"
#include <sqlite3.h>
#include <string>
#include <vector>
#include <optional>
#include <filesystem>
#include <memory>

namespace gemfinder {

namespace fs = std::filesystem;

struct FtsMatch {
    ChunkId chunk_id{0};
    double bm25_score{0.0};
    std::string snippet;
};

class Database {
public:
    explicit Database(const fs::path& db_path);
    ~Database();

    // Non-copyable, movable
    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;
    Database(Database&& other) noexcept;
    Database& operator=(Database&& other) noexcept;

    [[nodiscard]] bool isOpen() const noexcept { return db_ != nullptr; }

    // Schema initialization
    bool initializeSchema();

    // Transactions
    bool beginTransaction();
    bool commitTransaction();
    bool rollbackTransaction();

    // File operations
    std::optional<FileId> upsertFile(const FileRecord& file);
    std::optional<FileRecord> getFileById(FileId id);
    std::optional<FileRecord> getFileByPath(const fs::path& path);
    bool updateFileMtimeAndHash(FileId id, int64_t mtime, uint64_t hash);
    bool removeFile(FileId id);
    std::vector<FileRecord> getAllFiles();

    // Chunk operations
    std::optional<ChunkId> insertChunk(const ChunkRecord& chunk);
    std::optional<ChunkRecord> getChunkById(ChunkId id);
    std::vector<ChunkRecord> getChunksByFileId(FileId file_id);
    bool removeChunksByFileId(FileId file_id);

    // Full-Text Search (FTS5)
    std::vector<FtsMatch> searchFts(const std::string& query, size_t limit = 50);

    // Raw access for custom queries if needed
    [[nodiscard]] sqlite3* handle() const noexcept { return db_; }

private:
    sqlite3* db_{nullptr};
    fs::path path_;

    void close();
    bool execute(const char* sql);
};

} // namespace gemfinder
