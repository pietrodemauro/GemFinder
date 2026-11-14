#include "storage/Database.hpp"
#include "common/Logger.hpp"
#include <sstream>
#include <stdexcept>

namespace gemfinder {

Database::Database(const fs::path& db_path) : path_(db_path) {
    if (db_path != ":memory:") {
        if (auto parent = db_path.parent_path(); !parent.empty()) {
            fs::create_directories(parent);
        }
    }

    int rc = sqlite3_open(db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::string err = sqlite3_errmsg(db_);
        close();
        Logger::error("Impossibile aprire il database SQLite: " + err);
        return;
    }

    // Enable WAL mode and performance pragmas
    execute("PRAGMA journal_mode = WAL;");
    execute("PRAGMA synchronous = NORMAL;");
    execute("PRAGMA foreign_keys = ON;");
    execute("PRAGMA temp_store = MEMORY;");
    execute("PRAGMA cache_size = -64000;"); // 64MB cache
}

Database::~Database() {
    close();
}

Database::Database(Database&& other) noexcept : db_(other.db_), path_(std::move(other.path_)) {
    other.db_ = nullptr;
}

Database& Database::operator=(Database&& other) noexcept {
    if (this != &other) {
        close();
        db_ = other.db_;
        path_ = std::move(other.path_);
        other.db_ = nullptr;
    }
    return *this;
}

void Database::close() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool Database::execute(const char* sql) {
    char* errmsg = nullptr;
    int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        if (errmsg) {
            Logger::error("SQLite Exec Error: " + std::string(errmsg));
            sqlite3_free(errmsg);
        }
        return false;
    }
    return true;
}

bool Database::initializeSchema() {
    if (!db_) return false;

    const char* schema = R"(
        CREATE TABLE IF NOT EXISTS files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            extension TEXT,
            file_size INTEGER,
            mtime INTEGER NOT NULL,
            xxhash64 INTEGER,
            modality TEXT,
            status TEXT DEFAULT 'indexed'
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            chunk_type TEXT,
            start_offset INTEGER,
            end_offset INTEGER,
            preview_text TEXT,
            FOREIGN KEY(file_id) REFERENCES files(file_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
        CREATE INDEX IF NOT EXISTS idx_files_hash ON files(xxhash64);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            preview_text,
            tokenize = 'unicode61'
        );
    )";

    return execute(schema);
}

bool Database::beginTransaction() {
    return execute("BEGIN TRANSACTION;");
}

bool Database::commitTransaction() {
    return execute("COMMIT;");
}

bool Database::rollbackTransaction() {
    return execute("ROLLBACK;");
}

std::optional<FileId> Database::upsertFile(const FileRecord& file) {
    if (!db_) return std::nullopt;

    const char* sql = R"(
        INSERT INTO files (filepath, filename, extension, file_size, mtime, xxhash64, modality, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filepath) DO UPDATE SET
            filename = excluded.filename,
            extension = excluded.extension,
            file_size = excluded.file_size,
            mtime = excluded.mtime,
            xxhash64 = excluded.xxhash64,
            modality = excluded.modality,
            status = excluded.status
        RETURNING file_id;
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        Logger::error("SQLite Prepare Error (upsertFile): " + std::string(sqlite3_errmsg(db_)));
        return std::nullopt;
    }

    std::string filepath_str = file.filepath.string();
    std::string modality_str = modalityToString(file.modality);
    std::string status_str = statusToString(file.status);

    sqlite3_bind_text(stmt, 1, filepath_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, file.filename.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, file.extension.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 4, static_cast<sqlite3_int64>(file.file_size));
    sqlite3_bind_int64(stmt, 5, static_cast<sqlite3_int64>(file.mtime));
    sqlite3_bind_int64(stmt, 6, static_cast<sqlite3_int64>(file.xxhash64));
    sqlite3_bind_text(stmt, 7, modality_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 8, status_str.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<FileId> result = std::nullopt;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        result = sqlite3_column_int64(stmt, 0);
    } else {
        Logger::error("SQLite Step Error (upsertFile): " + std::string(sqlite3_errmsg(db_)));
    }

    sqlite3_finalize(stmt);
    return result;
}

std::optional<FileRecord> Database::getFileById(FileId id) {
    if (!db_) return std::nullopt;

    const char* sql = "SELECT file_id, filepath, filename, extension, file_size, mtime, xxhash64, modality, status FROM files WHERE file_id = ?;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;

    sqlite3_bind_int64(stmt, 1, id);

    std::optional<FileRecord> rec = std::nullopt;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        FileRecord f;
        f.id = sqlite3_column_int64(stmt, 0);
        f.filepath = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        f.filename = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        f.extension = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        f.file_size = static_cast<uint64_t>(sqlite3_column_int64(stmt, 4));
        f.mtime = sqlite3_column_int64(stmt, 5);
        f.xxhash64 = static_cast<uint64_t>(sqlite3_column_int64(stmt, 6));
        f.modality = stringToModality(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7)));
        if (const char* st = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 8))) {
            f.status = stringToStatus(st);
        }
        rec = f;
    }

    sqlite3_finalize(stmt);
    return rec;
}

std::optional<FileRecord> Database::getFileByPath(const fs::path& path) {
    if (!db_) return std::nullopt;

    const char* sql = "SELECT file_id, filepath, filename, extension, file_size, mtime, xxhash64, modality, status FROM files WHERE filepath = ?;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;

    std::string path_str = path.string();
    sqlite3_bind_text(stmt, 1, path_str.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<FileRecord> rec = std::nullopt;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        FileRecord f;
        f.id = sqlite3_column_int64(stmt, 0);
        f.filepath = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        f.filename = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        f.extension = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        f.file_size = static_cast<uint64_t>(sqlite3_column_int64(stmt, 4));
        f.mtime = sqlite3_column_int64(stmt, 5);
        f.xxhash64 = static_cast<uint64_t>(sqlite3_column_int64(stmt, 6));
        f.modality = stringToModality(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7)));
        if (const char* st = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 8))) {
            f.status = stringToStatus(st);
        }
        rec = f;
    }

    sqlite3_finalize(stmt);
    return rec;
}

bool Database::updateFileMtimeAndHash(FileId id, int64_t mtime, uint64_t hash) {
    if (!db_) return false;
    const char* sql = "UPDATE files SET mtime = ?, xxhash64 = ? WHERE file_id = ?;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

    sqlite3_bind_int64(stmt, 1, mtime);
    sqlite3_bind_int64(stmt, 2, static_cast<sqlite3_int64>(hash));
    sqlite3_bind_int64(stmt, 3, id);

    bool ok = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    return ok;
}

bool Database::removeFile(FileId id) {
    if (!db_) return false;
    // Chunks table has CASCADE on delete, but we also delete from FTS
    removeChunksByFileId(id);

    const char* sql = "DELETE FROM files WHERE file_id = ?;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

    sqlite3_bind_int64(stmt, 1, id);
    bool ok = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    return ok;
}

std::vector<FileRecord> Database::getAllFiles() {
    std::vector<FileRecord> files;
    if (!db_) return files;

    const char* sql = "SELECT file_id, filepath, filename, extension, file_size, mtime, xxhash64, modality, status FROM files ORDER BY file_id ASC;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return files;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        FileRecord f;
        f.id = sqlite3_column_int64(stmt, 0);
        f.filepath = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        f.filename = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        f.extension = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        f.file_size = static_cast<uint64_t>(sqlite3_column_int64(stmt, 4));
        f.mtime = sqlite3_column_int64(stmt, 5);
        f.xxhash64 = static_cast<uint64_t>(sqlite3_column_int64(stmt, 6));
        f.modality = stringToModality(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7)));
        if (const char* st = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 8))) {
            f.status = stringToStatus(st);
        }
        files.push_back(std::move(f));
    }

    sqlite3_finalize(stmt);
    return files;
}

std::optional<ChunkId> Database::insertChunk(const ChunkRecord& chunk) {
    if (!db_) return std::nullopt;

    const char* sql = R"(
        INSERT INTO chunks (file_id, chunk_type, start_offset, end_offset, preview_text)
        VALUES (?, ?, ?, ?, ?)
        RETURNING chunk_id;
    )";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        Logger::error("SQLite Prepare Error (insertChunk): " + std::string(sqlite3_errmsg(db_)));
        return std::nullopt;
    }

    sqlite3_bind_int64(stmt, 1, chunk.file_id);
    sqlite3_bind_text(stmt, 2, chunk.chunk_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 3, chunk.start_offset);
    sqlite3_bind_int64(stmt, 4, chunk.end_offset);
    sqlite3_bind_text(stmt, 5, chunk.preview_text.c_str(), -1, SQLITE_TRANSIENT);

    std::optional<ChunkId> result = std::nullopt;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        result = sqlite3_column_int64(stmt, 0);
    }

    sqlite3_finalize(stmt);

    if (result.has_value() && !chunk.preview_text.empty()) {
        const char* fts_sql = "INSERT INTO chunks_fts (rowid, preview_text) VALUES (?, ?);";
        sqlite3_stmt* fts_stmt = nullptr;
        if (sqlite3_prepare_v2(db_, fts_sql, -1, &fts_stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_int64(fts_stmt, 1, *result);
            sqlite3_bind_text(fts_stmt, 2, chunk.preview_text.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(fts_stmt);
            sqlite3_finalize(fts_stmt);
        }
    }

    return result;
}

std::optional<ChunkRecord> Database::getChunkById(ChunkId id) {
    if (!db_) return std::nullopt;

    const char* sql = "SELECT chunk_id, file_id, chunk_type, start_offset, end_offset, preview_text FROM chunks WHERE chunk_id = ?;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;

    sqlite3_bind_int64(stmt, 1, id);

    std::optional<ChunkRecord> rec = std::nullopt;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        ChunkRecord c;
        c.id = sqlite3_column_int64(stmt, 0);
        c.file_id = sqlite3_column_int64(stmt, 1);
        c.chunk_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        c.start_offset = sqlite3_column_int64(stmt, 3);
        c.end_offset = sqlite3_column_int64(stmt, 4);
        c.preview_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        rec = c;
    }

    sqlite3_finalize(stmt);
    return rec;
}

std::vector<ChunkRecord> Database::getChunksByFileId(FileId file_id) {
    std::vector<ChunkRecord> chunks;
    if (!db_) return chunks;

    const char* sql = "SELECT chunk_id, file_id, chunk_type, start_offset, end_offset, preview_text FROM chunks WHERE file_id = ? ORDER BY start_offset ASC;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return chunks;

    sqlite3_bind_int64(stmt, 1, file_id);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ChunkRecord c;
        c.id = sqlite3_column_int64(stmt, 0);
        c.file_id = sqlite3_column_int64(stmt, 1);
        c.chunk_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        c.start_offset = sqlite3_column_int64(stmt, 3);
        c.end_offset = sqlite3_column_int64(stmt, 4);
        c.preview_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        chunks.push_back(std::move(c));
    }

    sqlite3_finalize(stmt);
    return chunks;
}

bool Database::removeChunksByFileId(FileId file_id) {
    if (!db_) return false;

    // First delete from FTS index
    const char* fts_del_sql = "DELETE FROM chunks_fts WHERE rowid IN (SELECT chunk_id FROM chunks WHERE file_id = ?);";
    sqlite3_stmt* fts_stmt = nullptr;
    if (sqlite3_prepare_v2(db_, fts_del_sql, -1, &fts_stmt, nullptr) == SQLITE_OK) {
        sqlite3_bind_int64(fts_stmt, 1, file_id);
        sqlite3_step(fts_stmt);
        sqlite3_finalize(fts_stmt);
    }

    const char* sql = "DELETE FROM chunks WHERE file_id = ?;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

    sqlite3_bind_int64(stmt, 1, file_id);
    bool ok = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    return ok;
}

std::vector<FtsMatch> Database::searchFts(const std::string& query, size_t limit) {
    std::vector<FtsMatch> matches;
    if (!db_ || query.empty()) return matches;

    // FTS5 query using bm25 ranking and snippet highlighting
    const char* sql = R"(
        SELECT rowid, bm25(chunks_fts), snippet(chunks_fts, 0, '<b>', '</b>', '...', 16)
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY bm25(chunks_fts) ASC
        LIMIT ?;
    )";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        Logger::error("SQLite Prepare Error (searchFts): " + std::string(sqlite3_errmsg(db_)));
        return matches;
    }

    sqlite3_bind_text(stmt, 1, query.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, static_cast<sqlite3_int64>(limit));

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        FtsMatch m;
        m.chunk_id = sqlite3_column_int64(stmt, 0);
        m.bm25_score = sqlite3_column_double(stmt, 1);
        if (const char* snip = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2))) {
            m.snippet = snip;
        }
        matches.push_back(std::move(m));
    }

    sqlite3_finalize(stmt);
    return matches;
}

} // namespace gemfinder
