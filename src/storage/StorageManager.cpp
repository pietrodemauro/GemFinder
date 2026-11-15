#include "storage/StorageManager.hpp"
#include "common/Logger.hpp"
#include <unordered_map>
#include <algorithm>

namespace gemfinder {

StorageManager::StorageManager(const fs::path& db_path, const fs::path& vector_path, size_t vector_dim)
    : db_(db_path),
      vectors_(vector_dim),
      vector_path_(vector_path) {}

StorageManager::~StorageManager() {
    save();
}

bool StorageManager::initialize() {
    if (!db_.initializeSchema()) {
        Logger::error("Failed to initialize database schema.");
        return false;
    }

    if (fs::exists(vector_path_)) {
        if (!vectors_.load(vector_path_)) {
            Logger::warn("Could not load vector store from " + vector_path_.string() + ", creating fresh index.");
            vectors_.clear();
        } else {
            Logger::info("Loaded vector store with " + std::to_string(vectors_.size()) + " vectors.");
        }
    }

    return true;
}

bool StorageManager::save() {
    if (!vector_path_.empty()) {
        return vectors_.save(vector_path_);
    }
    return true;
}

bool StorageManager::indexFile(FileRecord file, const std::vector<std::pair<ChunkRecord, Embedding>>& chunks_with_embeddings) {
    if (!db_.beginTransaction()) {
        return false;
    }

    // 1. Remove previous chunks for this file from vector store if file already exists
    if (auto existing = db_.getFileByPath(file.filepath)) {
        auto old_chunks = db_.getChunksByFileId(existing->id);
        for (const auto& old_chunk : old_chunks) {
            vectors_.remove(old_chunk.id);
        }
        db_.removeChunksByFileId(existing->id);
    }

    // 2. Upsert file record
    file.status = IndexStatus::Indexed;
    auto file_id_opt = db_.upsertFile(file);
    if (!file_id_opt) {
        db_.rollbackTransaction();
        return false;
    }

    FileId file_id = *file_id_opt;

    // 3. Insert each chunk and its vector
    for (const auto& [chunk_meta, embedding] : chunks_with_embeddings) {
        ChunkRecord chunk = chunk_meta;
        chunk.file_id = file_id;

        auto chunk_id_opt = db_.insertChunk(chunk);
        if (!chunk_id_opt) {
            Logger::error("Failed to insert chunk for file: " + file.filename);
            db_.rollbackTransaction();
            return false;
        }

        ChunkId chunk_id = *chunk_id_opt;
        if (!vectors_.add(chunk_id, embedding)) {
            Logger::error("Failed to insert vector for chunk id " + std::to_string(chunk_id));
            db_.rollbackTransaction();
            return false;
        }
    }

    return db_.commitTransaction();
}

bool StorageManager::removeFile(const fs::path& path) {
    auto file_opt = db_.getFileByPath(path);
    if (!file_opt) return false;
    return removeFileById(file_opt->id);
}

bool StorageManager::removeFileById(FileId file_id) {
    auto chunks = db_.getChunksByFileId(file_id);
    for (const auto& c : chunks) {
        vectors_.remove(c.id);
    }
    return db_.removeFile(file_id);
}

size_t StorageManager::removeDirectory(const fs::path& dir_path) {
    std::error_code ec;
    fs::path canonical_target = fs::weakly_canonical(dir_path, ec);
    std::string target_str = ec ? dir_path.string() : canonical_target.string();
    if (!target_str.empty() && target_str.back() != fs::path::preferred_separator) {
        target_str += fs::path::preferred_separator;
    }

    auto all_files = db_.getAllFiles();
    std::vector<FileId> to_remove;
    for (const auto& f : all_files) {
        fs::path p = f.filepath;
        fs::path can_p = fs::weakly_canonical(p, ec);
        std::string p_str = ec ? p.string() : can_p.string();
        if (p_str.starts_with(target_str) || p_str == dir_path.string() || (!ec && can_p == canonical_target)) {
            to_remove.push_back(f.id);
        }
    }

    db_.beginTransaction();
    for (FileId fid : to_remove) {
        auto chunks = db_.getChunksByFileId(fid);
        for (const auto& c : chunks) {
            vectors_.remove(c.id);
        }
        db_.removeFile(fid);
    }
    db_.commitTransaction();
    save();

    return to_remove.size();
}

size_t StorageManager::totalIndexedFiles() {
    return db_.getAllFiles().size();
}

std::vector<SearchResult> StorageManager::hybridSearch(
    const std::optional<std::string>& text_query,
    const std::optional<std::span<const float>>& vector_query,
    size_t limit,
    double rrf_k,
    float min_similarity
) {
    std::vector<SearchResult> final_results;
    if (!text_query.has_value() && !vector_query.has_value()) {
        return final_results;
    }

    struct FusionCandidate {
        ChunkId chunk_id{0};
        int rank_vector{0};
        int rank_lexical{0};
        double rrf_score{0.0};
        float vector_similarity{0.0f};
    };

    std::unordered_map<ChunkId, FusionCandidate> candidate_map;

    // 1. Vector Search
    if (vector_query.has_value()) {
        auto vec_matches = vectors_.search(*vector_query, limit * 3);
        int valid_rank = 1;
        for (size_t i = 0; i < vec_matches.size(); ++i) {
            const auto& vm = vec_matches[i];
            if (min_similarity > 0.0f && vm.similarity < min_similarity) {
                continue;
            }
            int rank = valid_rank++;
            auto& cand = candidate_map[vm.chunk_id];
            cand.chunk_id = vm.chunk_id;
            cand.rank_vector = rank;
            cand.vector_similarity = vm.similarity;
            cand.rrf_score += 1.0 / (rrf_k + static_cast<double>(rank));
        }
    }

    // 2. Lexical Full-Text Search (FTS5)
    if (text_query.has_value() && !text_query->empty()) {
        auto fts_matches = db_.searchFts(*text_query, limit * 3);
        for (size_t i = 0; i < fts_matches.size(); ++i) {
            const auto& fm = fts_matches[i];
            int rank = static_cast<int>(i + 1);
            auto& cand = candidate_map[fm.chunk_id];
            cand.chunk_id = fm.chunk_id;
            cand.rank_lexical = rank;
            cand.rrf_score += 1.0 / (rrf_k + static_cast<double>(rank));
        }
    }

    if (candidate_map.empty()) {
        return final_results;
    }

    // 3. Collect and sort candidates by RRF score descending
    std::vector<FusionCandidate> sorted_candidates;
    sorted_candidates.reserve(candidate_map.size());
    for (auto& [id, cand] : candidate_map) {
        // If query was vector-only, fall back to pure vector similarity as score
        if (vector_query.has_value() && (!text_query.has_value() || text_query->empty())) {
            cand.rrf_score = static_cast<double>(cand.vector_similarity);
            if (min_similarity > 0.0f && cand.vector_similarity < min_similarity) {
                continue;
            }
        } else if (min_similarity > 0.0f) {
            // If candidate has vector match below threshold and no lexical match, skip it
            if (cand.rank_vector > 0 && cand.vector_similarity < min_similarity && cand.rank_lexical == 0) {
                continue;
            }
        }
        sorted_candidates.push_back(cand);
    }

    std::sort(sorted_candidates.begin(), sorted_candidates.end(), [](const FusionCandidate& a, const FusionCandidate& b) {
        return a.rrf_score > b.rrf_score;
    });

    // 4. Hydrate metadata from SQLite
    size_t count = std::min(limit, sorted_candidates.size());
    final_results.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        const auto& cand = sorted_candidates[i];
        auto chunk_opt = db_.getChunkById(cand.chunk_id);
        if (!chunk_opt) continue;

        auto file_opt = db_.getFileById(chunk_opt->file_id);
        if (!file_opt) continue;

        SearchResult sr;
        sr.file_id = file_opt->id;
        sr.chunk_id = chunk_opt->id;
        sr.filepath = file_opt->filepath.string();
        sr.filename = file_opt->filename;
        sr.preview_text = chunk_opt->preview_text;
        sr.chunk_type = chunk_opt->chunk_type;
        sr.start_offset = chunk_opt->start_offset;
        sr.score = cand.rrf_score;
        sr.similarity = cand.vector_similarity;
        sr.rank_vector = cand.rank_vector;
        sr.rank_lexical = cand.rank_lexical;

        final_results.push_back(std::move(sr));
    }

    return final_results;
}

} // namespace gemfinder
