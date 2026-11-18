#include "gui/SearchWorker.hpp"
#include "inference/EngineFactory.hpp"
#include "common/Logger.hpp"
#include "common/Paths.hpp"

#include <chrono>

namespace gemfinder {

SearchWorker::SearchWorker(std::optional<fs::path> model_path, int n_threads, bool use_gpu, QObject* parent)
    : QThread(parent), model_path_(std::move(model_path)), n_threads_(n_threads), use_gpu_(use_gpu) {
    checkDaemonStatus();
    start();
}

SearchWorker::~SearchWorker() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        abort_ = true;
    }
    wait();
}

void SearchWorker::setComputeDevice(bool use_gpu) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        use_gpu_ = use_gpu;
    }
    IpcRequest req;
    req.action = IpcAction::ConfigureEngine;
    req.use_gpu = use_gpu;
    req.model_path = model_path_.has_value() ? model_path_->string() : "";
    requestAction(req);
}

void SearchWorker::setModelPath(const fs::path& model_path) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        model_path_ = model_path;
    }
    IpcRequest req;
    req.action = IpcAction::ConfigureEngine;
    req.use_gpu = use_gpu_;
    req.model_path = model_path.string();
    requestAction(req);
}

bool SearchWorker::useGpu() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return use_gpu_;
}

std::string SearchWorker::currentModelName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (local_engine_) {
        return local_engine_->inference().modelName();
    }
    if (model_path_.has_value()) {
        return model_path_->filename().string();
    }
    return "embeddinggemma-2.gguf";
}

bool SearchWorker::isDaemonConnected() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return daemon_running_;
}

void SearchWorker::checkDaemonStatus() {
    bool current = ipc_client_.isDaemonRunning();
    if (current != daemon_running_) {
        daemon_running_ = current;
        emit daemonStatusChanged(daemon_running_);
    }
}

void SearchWorker::requestSearch(const IpcRequest& req) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_request_ = req;
    has_work_ = true;
}

IpcResponse SearchWorker::runLocalSearch(const IpcRequest& req) {
    IpcResponse resp;
    resp.success = false;

    // Use standard persistent DB path
    fs::path db_path = getDefaultDatabasePath();
    fs::path vec_path = getVectorStorePath(db_path);

    if (!local_engine_) {
        auto engine = EngineFactory::createEngine(model_path_, kEmbeddingDim, n_threads_, use_gpu_);
        local_engine_ = std::make_unique<SearchEngine>(db_path, vec_path, std::move(engine));
        if (!local_engine_->initialize()) {
            resp.error_message = "Unable to initialize local search engine.";
            local_engine_.reset();
            return resp;
        }
    }

    auto start_time = std::chrono::steady_clock::now();
    try {
        if (req.action == IpcAction::SearchText) {
            resp.results = local_engine_->search(req.query, req.limit, req.threshold);
            resp.success = true;
        } else if (req.action == IpcAction::SearchImage) {
            resp.results = local_engine_->searchByImage(req.path, req.limit, req.threshold);
            resp.success = true;
        } else if (req.action == IpcAction::SearchAudio) {
            resp.results = local_engine_->searchByAudio(req.path, req.limit, req.threshold);
            resp.success = true;
        } else if (req.action == IpcAction::IndexDirectory) {
            auto stats = local_engine_->indexDirectory(req.path);
            resp.total_files = stats.indexed_files;
            resp.total_chunks = stats.indexed_chunks;
            resp.skipped_files = stats.skipped_files;
            resp.success = true;
        } else if (req.action == IpcAction::IndexFile) {
            bool ok = local_engine_->indexSingleFile(req.path);
            if (ok) {
                local_engine_->save();
                resp.total_files = 1;
                resp.total_chunks = 1;
                resp.success = true;
            } else {
                resp.success = false;
                resp.error_message = "Failed to index file: " + req.path;
            }
        } else if (req.action == IpcAction::UnindexDirectory) {
            size_t removed = local_engine_->removeDirectory(req.path);
            resp.total_files = removed;
            resp.success = true;
        } else if (req.action == IpcAction::UnindexFile) {
            bool ok = local_engine_->removeFile(req.path);
            resp.success = ok;
            if (!ok) resp.error_message = "File not found in index.";
        } else if (req.action == IpcAction::ConfigureEngine) {
            use_gpu_ = req.use_gpu;
            if (!req.model_path.empty()) {
                model_path_ = req.model_path;
            }
            if (local_engine_) {
                auto new_inf = EngineFactory::createEngine(model_path_, kEmbeddingDim, n_threads_, use_gpu_);
                local_engine_->setInferenceEngine(std::move(new_inf));
                resp.model_name = local_engine_->inference().modelName();
            } else {
                resp.model_name = model_path_.has_value() ? model_path_->filename().string() : "Default";
            }
            resp.success = true;
        } else {
            resp.error_message = "Action not supported in local mode.";
        }
    } catch (const std::exception& e) {
        resp.error_message = e.what();
    }

    auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    resp.latency_ms = elapsed;
    return resp;
}

void SearchWorker::run() {
    while (true) {
        bool should_abort = false;
        bool work_to_do = false;
        IpcRequest req;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            should_abort = abort_;
            if (should_abort) break;

            if (has_work_) {
                req = pending_request_;
                has_work_ = false;
                work_to_do = true;
            }
        }

        if (work_to_do) {
            checkDaemonStatus();

            IpcResponse response;
            int timeout_ms = 4000;
            if (req.action == IpcAction::IndexDirectory || req.action == IpcAction::UnindexDirectory) {
                timeout_ms = 120000;
            } else if (req.action == IpcAction::IndexFile) {
                timeout_ms = 30000;
            } else if (req.action == IpcAction::ConfigureEngine) {
                timeout_ms = 30000;
            }
            if (daemon_running_) {
                auto opt_resp = ipc_client_.send(req, timeout_ms);
                if (opt_resp.has_value()) {
                    response = *opt_resp;
                } else {
                    checkDaemonStatus();
                    response = runLocalSearch(req);
                }
            } else {
                response = runLocalSearch(req);
            }
            response.action = req.action;

            if (response.success) {
                emit searchCompleted(response);
            } else {
                emit searchError(QString::fromStdString(response.error_message));
            }
        }

        msleep(50);
    }
}

} // namespace gemfinder
