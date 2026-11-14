#pragma once

#include "common/IpcProtocol.hpp"
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <atomic>

namespace gemfinder {

namespace fs = std::filesystem;

class IpcServer {
public:
    using RequestHandler = std::function<IpcResponse(const IpcRequest&)>;

    explicit IpcServer(
        fs::path socket_path = kDefaultSocketPath,
        RequestHandler handler = nullptr
    );
    ~IpcServer();

    bool start();
    void stop();

    [[nodiscard]] bool isRunning() const noexcept { return running_.load(); }
    [[nodiscard]] const fs::path& socketPath() const noexcept { return socket_path_; }

private:
    fs::path socket_path_;
    RequestHandler handler_;
    int server_fd_{-1};
    std::atomic<bool> running_{false};
    std::jthread accept_thread_;

    void runLoop();
    void handleClient(int client_fd);
};

} // namespace gemfinder
