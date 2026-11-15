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

    gemfinder::Logger::info("Starting " + std::string(gemfinder::kProjectName) +
                            "-daemon (v" + std::string(gemfinder::kVersion) + ")...");

    auto inference = gemfinder::EngineFactory::createEngine(model_path, gemfinder::kEmbeddingDim, n_threads);
    auto engine = std::make_unique<gemfinder::SearchEngine>(db_path, vec_path, std::move(inference));

    if (!engine->initialize()) {
        gemfinder::Logger::error("Failed to initialize search engine.");
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
                resp.results = engine->search(req.query, req.limit, req.threshold);
                resp.success = true;
                break;
            }
            case gemfinder::IpcAction::SearchImage: {
                resp.results = engine->searchByImage(req.path, req.limit, req.threshold);
                resp.success = true;
                break;
            }
            case gemfinder::IpcAction::SearchAudio: {
                resp.results = engine->searchByAudio(req.path, req.limit, req.threshold);
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
            case gemfinder::IpcAction::UnindexDirectory: {
                size_t removed = engine->removeDirectory(req.path);
                resp.total_files = removed;
                resp.success = true;
                break;
            }
            case gemfinder::IpcAction::UnindexFile: {
                bool ok = engine->removeFile(req.path);
                resp.success = ok;
                if (!ok) resp.error_message = "File not found in index";
                break;
            }
            case gemfinder::IpcAction::ConfigureEngine: {
                fs::path model_to_use = req.model_path.empty() ? engine->getStats().inference_model : req.model_path;
                auto new_inf = gemfinder::EngineFactory::createEngine(model_to_use, gemfinder::kEmbeddingDim, 0, req.use_gpu);
                engine->setInferenceEngine(std::move(new_inf));
                resp.model_name = engine->inference().modelName();
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
                resp.error_message = "Unsupported action";
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
            gemfinder::Logger::info("[WATCHER] Removed file: " + path.filename().string());
            engine->removeFile(path);
        } else {
            gemfinder::Logger::info("[WATCHER] Re-indexing file: " + path.filename().string());
            engine->indexSingleFile(path);
            engine->save();
        }
    });

    if (watcher.start()) {
        for (const auto& dir : watch_dirs) {
            if (fs::exists(dir)) {
                watcher.addWatch(dir);
                gemfinder::Logger::info("Watching directory: " + dir.string());
            }
        }
    }

    gemfinder::Logger::info("Daemon operational. Waiting for requests or filesystem events (Ctrl+C to stop)...");

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    gemfinder::Logger::info("Stopping daemon...");
    watcher.stop();
    ipc_server.stop();
    engine->save();
    gemfinder::Logger::info("Daemon stopped successfully.");
    return 0;
}
