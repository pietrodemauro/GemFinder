#include "common/Types.hpp"
#include "common/Version.hpp"
#include "common/Logger.hpp"
#include "common/Hasher.hpp"
#include "common/IpcProtocol.hpp"
#include "storage/Database.hpp"
#include "preprocessor/MediaPreprocessor.hpp"
#include "inference/EngineFactory.hpp"
#include "engine/SearchEngine.hpp"
#include "daemon/IpcClient.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;

void printUsage(std::string_view progName) {
    std::cout << progName << " - " << gemfinder::kProjectDescription << "\n\n"
              << "Usage:\n"
              << "  " << progName << " index <directory> [--model <model.gguf>] [--threads <N>] [--db <path>] [--force]\n"
              << "  " << progName << " unindex <directory> [--db <path>] Remove all files in directory from index\n"
              << "  " << progName << " search <query_text> [--limit <N>] [--threshold <0.0-1.0>] [--threads <N>] [--model <model.gguf>] [--db <path>]\n"
              << "  " << progName << " search --image <file.jpg> [--limit <N>] [--threshold <0.0-1.0>] [--threads <N>] [--model <model.gguf>] [--db <path>]\n"
              << "  " << progName << " search --audio <file.wav> [--limit <N>] [--threshold <0.0-1.0>] [--threads <N>] [--model <model.gguf>] [--db <path>]\n"
              << "  " << progName << " stats [--db <path>]\n"
              << "  " << progName << " list [--db <path>]       List all registered files in the index\n"
              << "  " << progName << " inspect <file>           Preprocess a file and inspect extracted chunks\n"
              << "  " << progName << " embed <text> [--model <model.gguf>] [--threads <N>] Compute embedding for text\n"
              << "  " << progName << " hash <file>              Compute xxHash64 for a file\n"
              << "  " << progName << " --info                   Show system info and loaded libraries\n"
              << "  " << progName << " --version                Show software version\n\n"
              << "Search & Resource Options:\n"
              << "  --threshold, -s <val> Minimum similarity threshold (0.0 to 1.0, e.g. 0.35)\n"
              << "  --threads, -t <N>    Manually set CPU worker threads (default: 50% of CPU cores)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string_view command = argv[1];

    if (command == "--version" || command == "-v") {
        std::cout << gemfinder::kProjectName << " version " << gemfinder::kVersion << "\n";
        return 0;
    }

    if (command == "--help" || command == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    if (command == "--info") {
        std::cout << "=== GemFinder System Info ===\n";
        std::cout << "Project:    " << gemfinder::kProjectName << " (" << gemfinder::kVersion << ")\n";
        std::cout << "C++ Std:    " << __cplusplus << "\n";
        std::cout << "SQLite:     " << sqlite3_libversion() << " (source: " << sqlite3_sourceid() << ")\n";
        std::cout << "Vector Dim: " << gemfinder::kEmbeddingDim << " (Matryoshka L2 Normalized)\n";
        std::cout << "Raw Dim:    " << gemfinder::kRawEmbeddingDim << "\n";
        return 0;
    }

    if ((command == "hash" || command == "--hash") && argc >= 3) {
        fs::path target = argv[2];
        if (!fs::exists(target)) {
            gemfinder::Logger::error("File not found: " + target.string());
            return 1;
        }
        uint64_t hash = gemfinder::Hasher::hashFile(target);
        std::cout << "File:   " << target << "\n";
        std::cout << "XXH64:  0x" << std::hex << hash << std::dec << " (" << hash << ")\n";
        return 0;
    }

    if ((command == "inspect" || command == "--inspect") && argc >= 3) {
        fs::path target = argv[2];
        if (!fs::exists(target)) {
            gemfinder::Logger::error("File not found: " + target.string());
            return 1;
        }

        gemfinder::MediaPreprocessor router;
        auto payload = router.processFile(target);
        if (!payload) {
            gemfinder::Logger::error("Unable to preprocess file: " + target.string());
            return 1;
        }

        std::cout << "=== GemFinder File Inspection ===\n";
        std::cout << "Path:      " << payload->file.filepath << "\n";
        std::cout << "Filename:  " << payload->file.filename << "\n";
        std::cout << "Modality:  " << gemfinder::modalityToString(payload->file.modality) << "\n";
        std::cout << "Size:      " << payload->file.file_size << " bytes\n";
        std::cout << "XXH64:     0x" << std::hex << payload->file.xxhash64 << std::dec << "\n";
        std::cout << "Chunks:    " << payload->chunks.size() << "\n\n";

        for (size_t i = 0; i < payload->chunks.size(); ++i) {
            const auto& chunk = payload->chunks[i];
            std::cout << "  [Chunk " << (i + 1) << "]\n";
            std::cout << "    Type:    " << chunk.record.chunk_type << "\n";
            std::cout << "    Offset:  " << chunk.record.start_offset << " -> " << chunk.record.end_offset << "\n";
            std::cout << "    Preview: " << chunk.record.preview_text << "\n";
            if (!chunk.raw_media_buffer.empty()) {
                std::cout << "    Tensor:  " << chunk.raw_media_buffer.size() << " floats\n";
            }
            std::cout << "    Prompt:  " << chunk.prompt.substr(0, std::min<size_t>(100, chunk.prompt.size()))
                      << (chunk.prompt.size() > 100 ? "..." : "") << "\n\n";
        }
        return 0;
    }

    if ((command == "embed" || command == "--embed") && argc >= 3) {
        std::string_view text = argv[2];
        std::optional<fs::path> model_path = std::nullopt;
        int n_threads = 0;
        for (int i = 3; i < argc; ++i) {
            if (std::string_view(argv[i]) == "--model" && i + 1 < argc) {
                model_path = argv[i + 1];
                i++;
            } else if ((std::string_view(argv[i]) == "--threads" || std::string_view(argv[i]) == "-t") && i + 1 < argc) {
                try { n_threads = std::stoi(argv[i + 1]); } catch (...) {}
                i++;
            }
        }

        auto start_t = std::chrono::high_resolution_clock::now();
        auto engine = gemfinder::EngineFactory::createEngine(model_path, gemfinder::kEmbeddingDim, n_threads);
        auto vec = engine->embedText(text);
        auto end_t = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();

        std::cout << "=== GemFinder Embedding Test ===\n";
        std::cout << "Engine:    " << engine->modelName() << "\n";
        std::cout << "Dim:       " << vec.size() << " floats (L2 normalized)\n";
        std::cout << "Latency:   " << elapsed_ms << " ms\n";
        std::cout << "Vector:    [";
        for (size_t i = 0; i < std::min<size_t>(8, vec.size()); ++i) {
            std::cout << (i > 0 ? ", " : "") << vec[i];
        }
        std::cout << ", ...]\n";
        return 0;
    }

    // Default paths for DB and Vector store
    fs::path default_db = "gemfinder_metadata.db";
    fs::path default_vec = "gemfinder_vectors.usearch";

    // Parse options: --db, --model, --limit, --force
    auto getOption = [&](std::string_view flag) -> std::optional<std::string> {
        for (int i = 2; i < argc; ++i) {
            if (std::string_view(argv[i]) == flag && i + 1 < argc) {
                return argv[i + 1];
            }
        }
        return std::nullopt;
    };

    auto hasFlag = [&](std::string_view flag) -> bool {
        for (int i = 2; i < argc; ++i) {
            if (std::string_view(argv[i]) == flag) return true;
        }
        return false;
    };

    fs::path db_path = getOption("--db").value_or(default_db.string());
    fs::path vec_path = db_path.parent_path() / (db_path.stem().string() + ".usearch");
    std::optional<fs::path> model_path = getOption("--model");

    int n_threads = 0;
    if (auto t_opt = getOption("--threads")) {
        try { n_threads = std::stoi(*t_opt); } catch (...) {}
    } else if (auto t_opt2 = getOption("-t")) {
        try { n_threads = std::stoi(*t_opt2); } catch (...) {}
    }

    // Command: index <directory>
    if (command == "index" && argc >= 3) {
        fs::path target_dir = argv[2];
        if (!fs::exists(target_dir) || !fs::is_directory(target_dir)) {
            gemfinder::Logger::error("Directory not found: " + target_dir.string());
            return 1;
        }

        bool force = hasFlag("--force");

        gemfinder::Logger::info("Initializing search engine for indexing...");
        auto inf = gemfinder::EngineFactory::createEngine(model_path, gemfinder::kEmbeddingDim, n_threads);
        gemfinder::SearchEngine engine(db_path, vec_path, std::move(inf));
        if (!engine.initialize()) {
            return 1;
        }

        std::cout << "\nScanning and indexing: " << target_dir << "\n";
        auto stats = engine.indexDirectory(target_dir, force, [](const std::string& cur, size_t i, size_t total) {
            std::cout << "\r[" << i << "/" << total << "] " << cur.substr(0, 50) << "              " << std::flush;
        });

        std::cout << "\n\n=== Indexing Results ===\n";
        std::cout << "Files scanned:      " << stats.scanned_files << "\n";
        std::cout << "Files indexed:      " << stats.indexed_files << "\n";
        std::cout << "Unmodified files:   " << stats.skipped_files << " (skipped instantly)\n";
        std::cout << "Chunks generated:   " << stats.indexed_chunks << "\n";
        std::cout << "Errors:             " << stats.error_files << "\n";
        std::cout << "Total elapsed time: " << std::fixed << std::setprecision(2) << stats.elapsed_seconds << " seconds\n";
        std::cout << "Database saved:     " << db_path << "\n";
        std::cout << "Vector index:       " << vec_path << "\n";
        return 0;
    }

    // Command: unindex <directory>
    if ((command == "unindex" || command == "remove-dir") && argc >= 3) {
        fs::path target_dir = argv[2];
        gemfinder::Logger::info("Initializing search engine for unindexing...");
        auto inf = gemfinder::EngineFactory::createEngine(model_path, gemfinder::kEmbeddingDim, n_threads);
        gemfinder::SearchEngine engine(db_path, vec_path, std::move(inf));
        if (!engine.initialize()) {
            return 1;
        }

        std::cout << "\nRemoving index for: " << target_dir << "\n";
        size_t removed = engine.removeDirectory(target_dir);
        std::cout << "Operation completed! Files removed from index: " << removed << "\n";
        return 0;
    }

    // Command: search <query> [--image <file>] [--audio <file>]
    if (command == "search") {
        size_t limit = 10;
        if (auto lim_str = getOption("--limit")) {
            try { limit = std::stoul(*lim_str); } catch (...) {}
        }

        float threshold = 0.0f;
        if (auto s_opt = getOption("--threshold")) {
            try { threshold = std::stof(*s_opt); } catch (...) {}
        } else if (auto s_opt2 = getOption("-s")) {
            try { threshold = std::stof(*s_opt2); } catch (...) {}
        }

        fs::path sock_path = getOption("--socket").value_or(std::string(gemfinder::kDefaultSocketPath));
        gemfinder::IpcClient client(sock_path);

        std::vector<gemfinder::SearchResult> results;
        double search_time_ms = 0.0;

        if (client.isDaemonRunning()) {
            gemfinder::IpcRequest req;
            req.limit = limit;
            req.threshold = threshold;

            if (auto img_path = getOption("--image")) {
                req.action = gemfinder::IpcAction::SearchImage;
                req.path = *img_path;
            } else if (auto aud_path = getOption("--audio")) {
                req.action = gemfinder::IpcAction::SearchAudio;
                req.path = *aud_path;
            } else if (argc >= 3) {
                req.action = gemfinder::IpcAction::SearchText;
                req.query = argv[2];
            } else {
                gemfinder::Logger::error("Specify a text query or --image <file> or --audio <file>");
                return 1;
            }

            auto start_t = std::chrono::high_resolution_clock::now();
            auto resp_opt = client.send(req);
            auto end_t = std::chrono::high_resolution_clock::now();
            search_time_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();

            if (resp_opt && resp_opt->success) {
                results = std::move(resp_opt->results);
            } else {
                gemfinder::Logger::error("Daemon IPC error: " + (resp_opt ? resp_opt->error_message : "timeout"));
                return 1;
            }
        } else {
            auto inf = gemfinder::EngineFactory::createEngine(model_path, gemfinder::kEmbeddingDim, n_threads);
            gemfinder::SearchEngine engine(db_path, vec_path, std::move(inf));
            if (!engine.initialize()) {
                return 1;
            }

            auto start_t = std::chrono::high_resolution_clock::now();
            if (auto img_path = getOption("--image")) {
                results = engine.searchByImage(*img_path, limit, threshold);
            } else if (auto aud_path = getOption("--audio")) {
                results = engine.searchByAudio(*aud_path, limit, threshold);
            } else if (argc >= 3) {
                results = engine.search(argv[2], limit, threshold);
            } else {
                gemfinder::Logger::error("Specify a text query or --image <file> or --audio <file>");
                return 1;
            }
            auto end_t = std::chrono::high_resolution_clock::now();
            search_time_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();
        }

        std::cout << "\n=== Search Results (" << results.size() << " found in "
                  << std::fixed << std::setprecision(2) << search_time_ms << " ms) ===\n\n";

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            std::cout << " " << (i + 1) << ". " << r.filename;
            if (r.chunk_type == "image_frame" || r.chunk_type == "audio_segment") {
                std::cout << "  [Similarity: " << std::fixed << std::setprecision(1) << (r.score * 100.0) << "%]\n";
            } else {
                std::cout << "  [Score RRF: " << std::setprecision(4) << r.score;
                if (r.similarity > 0.01f) {
                    std::cout << ", Similarity: " << std::fixed << std::setprecision(1) << (r.similarity * 100.0f) << "%";
                }
                std::cout << "]\n";
            }
            std::cout << "    Path:     " << r.filepath << "\n";
            std::cout << "    Offset:   ";
            if (r.chunk_type == "text_paragraph") {
                std::cout << "Line " << r.start_offset << "\n";
            } else if (r.chunk_type == "pdf_page") {
                std::cout << "Page " << r.start_offset << "\n";
            } else if (r.chunk_type == "audio_segment") {
                std::cout << "Minute " << (r.start_offset / 60) << ":" << (r.start_offset % 60) << "\n";
            } else {
                std::cout << r.chunk_type << "\n";
            }
            if (!r.preview_text.empty()) {
                std::cout << "    Snippet:  \"" << r.preview_text << "\"\n";
            }
            std::cout << "    Rank:     Vector: " << r.rank_vector << ", Lexical: " << r.rank_lexical << "\n\n";
        }

        return 0;
    }

    // Command: stats
    if (command == "stats") {
        auto inf = gemfinder::EngineFactory::createEngine(model_path, gemfinder::kEmbeddingDim, n_threads);
        gemfinder::SearchEngine engine(db_path, vec_path, std::move(inf));
        if (!engine.initialize()) {
            return 1;
        }

        auto stats = engine.getStats();
        std::cout << "=== GemFinder Index Statistics ===\n";
        std::cout << "Indexed Files:       " << stats.total_files << "\n";
        std::cout << "Total Chunks:        " << stats.total_chunks << "\n";
        std::cout << "USearch Vectors:     " << stats.vector_count << "\n";
        std::cout << "Vector Dimension:    " << stats.embedding_dim << " floats\n";
        std::cout << "Inference Model:     " << stats.inference_model << "\n";
        std::cout << "Metadata DB:         " << stats.db_path << "\n";
        std::cout << "Vector Index:        " << stats.vector_path << "\n";
        return 0;
    }

    // Command: list
    if (command == "list") {
        gemfinder::Database db(db_path);
        if (!db.initializeSchema()) {
            std::cerr << "Unable to open database: " << db_path << "\n";
            return 1;
        }

        auto files = db.getAllFiles();
        std::cout << "\n=== GemFinder Registered Files (" << files.size() << " files total) ===\n\n";
        if (files.empty()) {
            std::cout << "No files present in database.\n";
            return 0;
        }

        size_t indexed_count = 0;
        size_t error_count = 0;
        size_t pending_count = 0;

        for (size_t i = 0; i < files.size(); ++i) {
            const auto& f = files[i];
            std::string mod_icon = "📄";
            if (f.modality == gemfinder::Modality::Image) mod_icon = "🖼️";
            else if (f.modality == gemfinder::Modality::Audio) mod_icon = "🎵";

            std::string status_badge = "[Indexed]";
            if (f.status == gemfinder::IndexStatus::Error) {
                status_badge = "[Error]";
                error_count++;
            } else if (f.status == gemfinder::IndexStatus::Pending) {
                status_badge = "[Pending]";
                pending_count++;
            } else {
                indexed_count++;
            }

            std::cout << std::setw(4) << (i + 1) << ". " << mod_icon << " " << f.filename
                      << "  " << status_badge
                      << " (" << (f.file_size / 1024) << " KB)\n";
            std::cout << "       Path: " << f.filepath.string() << "\n";
        }

        std::cout << "\nSummary: " << indexed_count << " indexed, "
                  << pending_count << " pending, "
                  << error_count << " with errors.\n\n";
        return 0;
    }

    std::cerr << "Unknown command: " << command << "\n";
    printUsage(argv[0]);
    return 1;
}
