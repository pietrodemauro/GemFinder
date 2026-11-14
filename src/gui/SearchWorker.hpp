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
    explicit SearchWorker(std::optional<fs::path> model_path = std::nullopt, QObject* parent = nullptr);
    ~SearchWorker() override;

    void requestSearch(const IpcRequest& req);
    [[nodiscard]] bool isDaemonConnected() const;

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
    IpcClient ipc_client_;
    std::unique_ptr<SearchEngine> local_engine_;
    bool daemon_running_{false};
    bool abort_{false};
    bool has_work_{false};
    IpcRequest pending_request_;
};

} // namespace gemfinder
