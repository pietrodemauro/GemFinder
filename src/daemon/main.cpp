#include "common/Version.hpp"
#include "common/Logger.hpp"
#include "common/IpcProtocol.hpp"
#include "daemon/IpcServer.hpp"
#include "daemon/FileWatcher.hpp"
#include "engine/SearchEngine.hpp"
#include "inference/EngineFactory.hpp"

#include <iostream>
#include <csignal>
#include <atomic>
#include <chrono>

namespace fs = std::filesystem;

static std::atomic<bool> g_running{true};

void signalHandler(int signum) {
    (void)signum;
    g_running.store(false);
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    fs::path db_path = "gemfinder_metadata.db";
    fs::path socket_path = gemfinder::kDefaultSocketPath;
    std::optional<fs::path> model_path = std::nullopt;
    std::vector<fs::path> watch_dirs;
    int n_threads = 0;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--db" && i + 1 < argc) {
            db_path = argv[++i];
        } else if (arg == "--socket" && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) {
            try { n_threads = std::stoi(argv[++i]); } catch (...) {}
        } else if (arg == "--watch" && i + 1 < argc) {
            watch_dirs.emplace_back(argv[++i]);
        }
    }

    fs::path vec_path = db_path.parent_path() / (db_path.stem().string() + ".usearch");

    gemfinder::Logger::info("Avvio di " + std::string(gemfinder::kProjectName) +
                            "-daemon (v" + std::string(gemfinder::kVersion) + ")...");

    auto inference = gemfinder::EngineFactory::createEngine(model_path, gemfinder::kEmbeddingDim, n_threads);
    auto engine = std::make_unique<gemfinder::SearchEngine>(db_path, vec_path, std::move(inference));

    if (!engine->initialize()) {
        gemfinder::Logger::error("Impossibile inizializzare il motore di ricerca.");
        return 1;
    }

    // IPC Request Handler
    auto handler = [&](const gemfinder::IpcRequest& req) -> gemfinder::IpcResponse {
        gemfinder::IpcResponse resp;
        auto start_t = std::chrono::high_resolution_clock::now();

        switch (req.action) {
            case gemfinder::IpcAction::Ping: {
                resp.success = true;
                break;
            }
            case gemfinder::IpcAction::SearchText: {
                resp.results = engine->search(req.query, req.limit);
                resp.success = true;
                break;
            }
            case gemfinder::IpcAction::SearchImage: {
                resp.results = engine->searchByImage(req.path, req.limit);
                resp.success = true;
                break;
            }
            case gemfinder::IpcAction::SearchAudio: {
                resp.results = engine->searchByAudio(req.path, req.limit);
                resp.success = true;
                break;
            }
            case gemfinder::IpcAction::IndexDirectory: {
                auto stats = engine->indexDirectory(req.path);
                resp.total_files = stats.indexed_files;
                resp.total_chunks = stats.indexed_chunks;
                resp.success = true;
                break;
            }
            case gemfinder::IpcAction::GetStats: {
                auto s = engine->getStats();
                resp.total_files = s.total_files;
                resp.total_chunks = s.total_chunks;
                resp.model_name = s.inference_model;
                resp.success = true;
                break;
            }
            default: {
                resp.success = false;
                resp.error_message = "Azione non supportata";
                break;
            }
        }

        auto end_t = std::chrono::high_resolution_clock::now();
        resp.latency_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();
        return resp;
    };

    // Initialize IPC Server
    gemfinder::IpcServer ipc_server(socket_path, handler);
    if (!ipc_server.start()) {
        return 1;
    }

    // Initialize File Watcher with debouncing
    gemfinder::FileWatcher watcher([&](const fs::path& path, gemfinder::FileEvent ev) {
        if (ev == gemfinder::FileEvent::Deleted) {
            gemfinder::Logger::info("[WATCHER] Rimosso file: " + path.filename().string());
            engine->removeFile(path);
        } else {
            gemfinder::Logger::info("[WATCHER] Re-indicizzazione: " + path.filename().string());
            engine->indexSingleFile(path);
            engine->save();
        }
    });

    if (watcher.start()) {
        for (const auto& dir : watch_dirs) {
            if (fs::exists(dir)) {
                watcher.addWatch(dir);
                gemfinder::Logger::info("Monitoraggio cartella: " + dir.string());
            }
        }
    }

    gemfinder::Logger::info("Demone operativo. In attesa di richieste o eventi filesystem (Ctrl+C per arrestare)...");

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    gemfinder::Logger::info("Arresto del demone in corso...");
    watcher.stop();
    ipc_server.stop();
    engine->save();
    gemfinder::Logger::info("Demone arrestato con successo.");
    return 0;
}
