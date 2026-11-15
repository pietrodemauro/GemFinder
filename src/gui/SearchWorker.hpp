#pragma once

#include "common/IpcProtocol.hpp"
#include "daemon/IpcClient.hpp"
#include "engine/SearchEngine.hpp"
#include <QThread>
#include <QString>
#include <memory>
#include <mutex>

namespace gemfinder {

class SearchWorker : public QThread {
    Q_OBJECT

public:
    explicit SearchWorker(std::optional<fs::path> model_path = std::nullopt, int n_threads = 0, bool use_gpu = true, QObject* parent = nullptr);
    ~SearchWorker() override;

    void requestSearch(const IpcRequest& req);
    void requestAction(const IpcRequest& req) { requestSearch(req); }
    void setComputeDevice(bool use_gpu);
    void setModelPath(const fs::path& model_path);
    [[nodiscard]] bool isDaemonConnected() const;
    [[nodiscard]] bool useGpu() const;
    [[nodiscard]] std::string currentModelName() const;

signals:
    void searchCompleted(const IpcResponse& response);
    void searchError(const QString& error_msg);
    void daemonStatusChanged(bool is_running);

protected:
    void run() override;

private:
    void checkDaemonStatus();
    IpcResponse runLocalSearch(const IpcRequest& req);

    mutable std::mutex mutex_;
    std::optional<fs::path> model_path_;
    int n_threads_{0};
    bool use_gpu_{true};
    IpcClient ipc_client_;
    std::unique_ptr<SearchEngine> local_engine_;
    bool daemon_running_{false};
    bool abort_{false};
    bool has_work_{false};
    IpcRequest pending_request_;
};

} // namespace gemfinder
