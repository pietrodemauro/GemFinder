#include "common/Version.hpp"
#include "common/Logger.hpp"
#include "common/Hasher.hpp"
#include "common/Types.hpp"
#include "preprocessor/MediaPreprocessor.hpp"
#include "inference/EngineFactory.hpp"
#include "engine/SearchEngine.hpp"
#include "daemon/IpcClient.hpp"

#include <sqlite3.h>
#include <iostream>
#include <string_view>
#include <filesystem>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;

void printUsage(std::string_view progName) {
    std::cout << progName << " - " << gemfinder::kProjectDescription << "\n\n"
              << "Uso:\n"
              << "  " << progName << " index <cartella> [--model <modello.gguf>] [--db <path>] [--force]\n"
              << "  " << progName << " search <query_testo> [--limit <N>] [--model <modello.gguf>] [--db <path>]\n"
              << "  " << progName << " search --image <file.jpg> [--limit <N>] [--model <modello.gguf>] [--db <path>]\n"
              << "  " << progName << " search --audio <file.wav> [--limit <N>] [--model <modello.gguf>] [--db <path>]\n"
              << "  " << progName << " stats [--db <path>]\n"
              << "  " << progName << " list [--db <path>]       Elenca tutti i file registrati nell'indice\n"
              << "  " << progName << " inspect <file>           Pre-processa un file e mostra i chunk estratti\n"
              << "  " << progName << " embed <testo> [--model <modello.gguf>] Calcola l'embedding di un testo\n"
              << "  " << progName << " hash <file>              Calcola l'hash xxHash64 di un file\n"
              << "  " << progName << " --info                   Mostra le librerie e lo stato del sistema\n"
              << "  " << progName << " --version                Mostra la versione del software\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string_view command = argv[1];

    if (command == "--version" || command == "-v") {
        std::cout << gemfinder::kProjectName << " versione " << gemfinder::kVersion << "\n";
        return 0;
    }

    if (command == "--help" || command == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    if (command == "--info") {
        std::cout << "=== GemFinder System Info ===\n";
        std::cout << "Progetto:   " << gemfinder::kProjectName << " (" << gemfinder::kVersion << ")\n";
        std::cout << "C++ Std:    " << __cplusplus << "\n";
        std::cout << "SQLite:     " << sqlite3_libversion() << " (source: " << sqlite3_sourceid() << ")\n";
        std::cout << "Vector Dim: " << gemfinder::kEmbeddingDim << " (Matryoshka L2 Normalizzato)\n";
        std::cout << "Raw Dim:    " << gemfinder::kRawEmbeddingDim << "\n";
        return 0;
    }

    if ((command == "hash" || command == "--hash") && argc >= 3) {
        fs::path target = argv[2];
        if (!fs::exists(target)) {
            gemfinder::Logger::error("File non trovato: " + target.string());
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
            gemfinder::Logger::error("File non trovato: " + target.string());
            return 1;
        }

        gemfinder::MediaPreprocessor router;
        auto payload = router.processFile(target);
        if (!payload) {
            gemfinder::Logger::error("Impossibile pre-processare il file: " + target.string());
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
            std::cout << "    Tipo:    " << chunk.record.chunk_type << "\n";
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
        for (int i = 3; i < argc; ++i) {
            if (std::string_view(argv[i]) == "--model" && i + 1 < argc) {
                model_path = argv[i + 1];
                i++;
            }
        }

        auto start_t = std::chrono::high_resolution_clock::now();
        auto engine = gemfinder::EngineFactory::createEngine(model_path);
        auto vec = engine->embedText(text);
        auto end_t = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();

        std::cout << "=== GemFinder Embedding Test ===\n";
        std::cout << "Engine:    " << engine->modelName() << "\n";
        std::cout << "Dim:       " << vec.size() << " floats (L2 normalizzato)\n";
        std::cout << "Latency:   " << elapsed_ms << " ms\n";
        std::cout << "Vector:    [";
        for (size_t i = 0; i < std::min<size_t>(8, vec.size()); ++i) {
            std::cout << (i > 0 ? ", " : "") << vec[i];
        }
        std::cout << ", ...]\n";
        return 0;
    }

    // Default paths for DB and Vector store in current directory or ~/.local/share/gemfinder
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

    // Command: index <directory>
    if (command == "index" && argc >= 3) {
        fs::path target_dir = argv[2];
        if (!fs::exists(target_dir) || !fs::is_directory(target_dir)) {
            gemfinder::Logger::error("Cartella non trovata: " + target_dir.string());
            return 1;
        }

        bool force = hasFlag("--force");

        gemfinder::Logger::info("Inizializzazione motore di ricerca per l'indicizzazione...");
        auto inf = gemfinder::EngineFactory::createEngine(model_path);
        gemfinder::SearchEngine engine(db_path, vec_path, std::move(inf));
        if (!engine.initialize()) {
            return 1;
        }

        std::cout << "\nScansione e indicizzazione di: " << target_dir << "\n";
        auto stats = engine.indexDirectory(target_dir, force, [](const std::string& cur, size_t i, size_t total) {
            std::cout << "\r[" << i << "/" << total << "] " << cur.substr(0, 50) << "              " << std::flush;
        });

        std::cout << "\n\n=== Risultati Indicizzazione ===\n";
        std::cout << "File scansionati:   " << stats.scanned_files << "\n";
        std::cout << "File indicizzati:   " << stats.indexed_files << "\n";
        std::cout << "File non modificati:" << stats.skipped_files << " (saltati istantaneamente)\n";
        std::cout << "Chunk generati:     " << stats.indexed_chunks << "\n";
        std::cout << "Errori:             " << stats.error_files << "\n";
        std::cout << "Tempo totale:       " << std::fixed << std::setprecision(2) << stats.elapsed_seconds << " secondi\n";
        std::cout << "Database salvato:   " << db_path << "\n";
        std::cout << "Indice vettoriale:  " << vec_path << "\n";
        return 0;
    }

    // Command: search <query> [--image <file>] [--audio <file>]
    if (command == "search") {
        size_t limit = 10;
        if (auto lim_str = getOption("--limit")) {
            try { limit = std::stoul(*lim_str); } catch (...) {}
        }

        fs::path sock_path = getOption("--socket").value_or(std::string(gemfinder::kDefaultSocketPath));
        gemfinder::IpcClient client(sock_path);

        std::vector<gemfinder::SearchResult> results;
        double search_time_ms = 0.0;

        if (client.isDaemonRunning()) {
            gemfinder::IpcRequest req;
            req.limit = limit;

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
                gemfinder::Logger::error("Specificare una query testuale o --image <file> o --audio <file>");
                return 1;
            }

            auto start_t = std::chrono::high_resolution_clock::now();
            auto resp_opt = client.send(req);
            auto end_t = std::chrono::high_resolution_clock::now();
            search_time_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();

            if (resp_opt && resp_opt->success) {
                results = std::move(resp_opt->results);
            } else {
                gemfinder::Logger::error("Errore IPC dal demone: " + (resp_opt ? resp_opt->error_message : "timeout"));
                return 1;
            }
        } else {
            auto inf = gemfinder::EngineFactory::createEngine(model_path);
            gemfinder::SearchEngine engine(db_path, vec_path, std::move(inf));
            if (!engine.initialize()) {
                return 1;
            }

            auto start_t = std::chrono::high_resolution_clock::now();
            if (auto img_path = getOption("--image")) {
                results = engine.searchByImage(*img_path, limit);
            } else if (auto aud_path = getOption("--audio")) {
                results = engine.searchByAudio(*aud_path, limit);
            } else if (argc >= 3) {
                results = engine.search(argv[2], limit);
            } else {
                gemfinder::Logger::error("Specificare una query testuale o --image <file> o --audio <file>");
                return 1;
            }
            auto end_t = std::chrono::high_resolution_clock::now();
            search_time_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();
        }

        std::cout << "\n=== Risultati Ricerca (" << results.size() << " trovati in "
                  << std::fixed << std::setprecision(2) << search_time_ms << " ms) ===\n\n";

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            std::cout << " " << (i + 1) << ". " << r.filename
                      << "  [Score RRF: " << std::setprecision(4) << r.score << "]\n";
            std::cout << "    Percorso: " << r.filepath << "\n";
            std::cout << "    Punto:    ";
            if (r.chunk_type == "text_paragraph") {
                std::cout << "Riga " << r.start_offset << "\n";
            } else if (r.chunk_type == "audio_segment") {
                std::cout << "Minuto " << (r.start_offset / 60) << ":" << (r.start_offset % 60) << "\n";
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
        auto inf = gemfinder::EngineFactory::createEngine(model_path);
        gemfinder::SearchEngine engine(db_path, vec_path, std::move(inf));
        if (!engine.initialize()) {
            return 1;
        }

        auto stats = engine.getStats();
        std::cout << "=== GemFinder Index Statistics ===\n";
        std::cout << "File indicizzati:    " << stats.total_files << "\n";
        std::cout << "Chunk totali:        " << stats.total_chunks << "\n";
        std::cout << "Vettori in USearch:  " << stats.vector_count << "\n";
        std::cout << "Dimensione Vettori:  " << stats.embedding_dim << " floats\n";
        std::cout << "Modello Inferenza:   " << stats.inference_model << "\n";
        std::cout << "DB Metadati:         " << stats.db_path << "\n";
        std::cout << "Indice Vettoriale:   " << stats.vector_path << "\n";
        return 0;
    }

    // Command: list
    if (command == "list") {
        gemfinder::Database db(db_path);
        if (!db.initializeSchema()) {
            std::cerr << "Impossibile aprire il database: " << db_path << "\n";
            return 1;
        }

        auto files = db.getAllFiles();
        std::cout << "\n=== GemFinder File Registrati (" << files.size() << " file totali) ===\n\n";
        if (files.empty()) {
            std::cout << "Nessun file presente nel database.\n";
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
            std::cout << "       Percorso: " << f.filepath.string() << "\n";
        }

        std::cout << "\nRiepilogo: " << indexed_count << " indicizzati, "
                  << pending_count << " in attesa, "
                  << error_count << " con errori.\n\n";
        return 0;
    }

    std::cerr << "Comando sconosciuto: " << command << "\n";
    printUsage(argv[0]);
    return 1;
}
