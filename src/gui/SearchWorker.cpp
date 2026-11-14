#include "gui/SearchWorker.hpp"
#include "inference/EngineFactory.hpp"
#include "common/Logger.hpp"

#include <chrono>

namespace gemfinder {

SearchWorker::SearchWorker(QObject* parent)
    : QThread(parent) {
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

    // Check if local DB exists
    fs::path db_path = "gemfinder_metadata.db";
    if (!fs::exists(db_path) && fs::exists("gemfinder.db")) {
        db_path = "gemfinder.db";
    }
    fs::path vec_path = db_path.parent_path() / (db_path.stem().string() + ".usearch");
    if (!fs::exists(db_path)) {
        resp.error_message = "Demone non in esecuzione e nessun database locale trovato (gemfinder_metadata.db).";
        return resp;
    }

    if (!local_engine_) {
        auto engine = EngineFactory::createEngine(std::nullopt, kEmbeddingDim);
        local_engine_ = std::make_unique<SearchEngine>(db_path, vec_path, std::move(engine));
        if (!local_engine_->initialize()) {
            resp.error_message = "Impossibile inizializzare il motore di ricerca locale.";
            local_engine_.reset();
            return resp;
        }
    }

    auto start_time = std::chrono::steady_clock::now();
    try {
        if (req.action == IpcAction::SearchText) {
            resp.results = local_engine_->search(req.query, req.limit);
            resp.success = true;
        } else if (req.action == IpcAction::SearchImage) {
            resp.results = local_engine_->searchByImage(req.path, req.limit);
            resp.success = true;
        } else if (req.action == IpcAction::SearchAudio) {
            resp.results = local_engine_->searchByAudio(req.path, req.limit);
            resp.success = true;
        } else {
            resp.error_message = "Azione non supportata in modalità locale.";
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
            if (daemon_running_) {
                auto opt_resp = ipc_client_.send(req, 4000);
                if (opt_resp.has_value()) {
                    response = *opt_resp;
                } else {
                    checkDaemonStatus();
                    response = runLocalSearch(req);
                }
            } else {
                response = runLocalSearch(req);
            }

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
