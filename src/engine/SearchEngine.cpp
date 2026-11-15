#include "engine/SearchEngine.hpp"
#include "common/Logger.hpp"
#include "common/Hasher.hpp"
#include <chrono>

namespace gemfinder {

SearchEngine::SearchEngine(
    const fs::path& db_path,
    const fs::path& vector_path,
    std::unique_ptr<InferenceEngine> inference_engine
) : storage_(std::make_unique<StorageManager>(db_path, vector_path, inference_engine ? inference_engine->dimension() : kEmbeddingDim)),
    inference_(std::move(inference_engine)),
    db_path_(db_path),
    vector_path_(vector_path) {}

SearchEngine::~SearchEngine() {
    save();
}

bool SearchEngine::initialize() {
    if (!storage_->initialize()) {
        Logger::error("SearchEngine: Failed to initialize StorageManager.");
        return false;
    }
    return true;
}

bool SearchEngine::save() {
    return storage_->save();
}

bool SearchEngine::indexSingleFile(const fs::path& path) {
    try {
        auto payload_opt = preprocessor_.processFile(path);
        if (!payload_opt) {
            return false;
        }

        auto& payload = *payload_opt;
        std::vector<std::pair<ChunkRecord, Embedding>> chunks_with_embeddings;
        chunks_with_embeddings.reserve(payload.chunks.size());

        for (const auto& prepared : payload.chunks) {
            Embedding emb;

            if (payload.file.modality == Modality::Image && !prepared.raw_media_buffer.empty()) {
                emb = inference_->embedImage(prepared.raw_media_buffer, 224, 224);
            } else if (payload.file.modality == Modality::Audio && !prepared.raw_media_buffer.empty()) {
                emb = inference_->embedAudio(prepared.raw_media_buffer, 16000);
            } else {
                emb = inference_->embedText(prepared.prompt);
            }

            chunks_with_embeddings.emplace_back(prepared.record, std::move(emb));
        }

        return storage_->indexFile(payload.file, chunks_with_embeddings);
    } catch (const std::exception& e) {
        Logger::error("SearchEngine: Error indexing " + path.string() + ": " + e.what());
        return false;
    } catch (...) {
        Logger::error("SearchEngine: Unknown error indexing " + path.string());
        return false;
    }
}

bool SearchEngine::removeFile(const fs::path& path) {
    return storage_->removeFile(path);
}

size_t SearchEngine::removeDirectory(const fs::path& dir_path) {
    size_t removed = storage_->removeDirectory(dir_path);
    Logger::info("SearchEngine: Removed " + std::to_string(removed) + " files from index for directory: " + dir_path.string());
    return removed;
}

IndexStats SearchEngine::indexDirectory(
    const fs::path& root_path,
    bool force_reindex,
    const std::function<void(const std::string&, size_t, size_t)>& on_progress
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    IndexStats stats;

    auto files = crawler_.crawl(root_path);
    stats.scanned_files = files.size();

    size_t processed_count = 0;
    for (const auto& file_path : files) {
        processed_count++;
        if (on_progress) {
            on_progress(file_path.filename().string(), processed_count, files.size());
        }

        std::error_code ec;
        uint64_t size = fs::file_size(file_path, ec);
        if (ec) {
            stats.error_files++;
            continue;
        }

        int64_t mtime = getFileMtime(file_path);

        // Check if file is already indexed, unmodified, and its vectors are present in vector store
        if (!force_reindex) {
            auto existing = storage_->database().getFileByPath(file_path);
            if (existing.has_value()) {
                bool vectors_present = true;
                auto chunks = storage_->database().getChunksByFileId(existing->id);
                if (chunks.empty()) {
                    vectors_present = false;
                } else {
                    for (const auto& c : chunks) {
                        if (!storage_->vectorStore().contains(c.id)) {
                            vectors_present = false;
                            break;
                        }
                    }
                }

                if (vectors_present && existing->file_size == size && existing->mtime == mtime && existing->status == IndexStatus::Indexed) {
                    stats.skipped_files++;
                    continue;
                }
            }
        }

        // Index the file
        if (indexSingleFile(file_path)) {
            stats.indexed_files++;
            auto chunks = storage_->database().getChunksByFileId(
                storage_->database().getFileByPath(file_path)->id
            );
            stats.indexed_chunks += chunks.size();
        } else {
            stats.error_files++;
        }
    }

    save();

    auto end_time = std::chrono::high_resolution_clock::now();
    stats.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    return stats;
}

std::vector<SearchResult> SearchEngine::search(
    std::string_view query_text,
    size_t limit,
    float min_similarity,
    double rrf_k
) {
    std::string formatted_query = TextPreprocessor::formatQueryPrompt(query_text);
    Embedding query_vec = inference_->embedText(formatted_query);

    return storage_->hybridSearch(
        std::string(query_text),
        std::span<const float>(query_vec),
        limit,
        rrf_k,
        min_similarity
    );
}

std::vector<SearchResult> SearchEngine::searchByImage(
    const fs::path& image_path,
    size_t limit,
    float min_similarity
) {
    auto processed_opt = preprocessor_.image().processFile(image_path);
    if (!processed_opt) {
        Logger::error("SearchEngine: Unable to load query image: " + image_path.string());
        return {};
    }

    Embedding query_vec = inference_->embedImage(
        processed_opt->rgb_planar_or_chw,
        processed_opt->target_width,
        processed_opt->target_height
    );

    return storage_->hybridSearch(
        std::nullopt,
        std::span<const float>(query_vec),
        limit,
        60.0,
        min_similarity
    );
}

std::vector<SearchResult> SearchEngine::searchByAudio(
    const fs::path& audio_path,
    size_t limit,
    float min_similarity
) {
    auto processed_opt = preprocessor_.audio().processFile(audio_path);
    if (!processed_opt || processed_opt->chunks.empty()) {
        Logger::error("SearchEngine: Unable to load query audio: " + audio_path.string());
        return {};
    }

    // Embed the first segment of the query audio
    Embedding query_vec = inference_->embedAudio(
        processed_opt->chunks[0].pcm_samples,
        processed_opt->sample_rate
    );

    return storage_->hybridSearch(
        std::nullopt,
        std::span<const float>(query_vec),
        limit,
        60.0,
        min_similarity
    );
}

EngineStats SearchEngine::getStats() {
    EngineStats stats;
    stats.total_files = storage_->totalIndexedFiles();
    stats.total_chunks = storage_->totalIndexedChunks();
    stats.vector_count = storage_->vectorStore().size();
    stats.embedding_dim = inference_->dimension();
    stats.inference_model = inference_->modelName();
    stats.db_path = db_path_;
    stats.vector_path = vector_path_;
    return stats;
}

void SearchEngine::setInferenceEngine(std::unique_ptr<InferenceEngine> new_engine) {
    if (new_engine) {
        inference_ = std::move(new_engine);
    }
}

} // namespace gemfinder
