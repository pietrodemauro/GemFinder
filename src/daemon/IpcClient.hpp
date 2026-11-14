#pragma once

#include "common/IpcProtocol.hpp"
#include <filesystem>
#include <optional>

namespace gemfinder {

namespace fs = std::filesystem;

class IpcClient {
public:
    explicit IpcClient(fs::path socket_path = kDefaultSocketPath);

    // Send request and wait for response
    [[nodiscard]] std::optional<IpcResponse> send(const IpcRequest& request, int timeout_ms = 5000) const;

    // Check if the daemon is alive (ping)
    [[nodiscard]] bool isDaemonRunning() const;

private:
    fs::path socket_path_;
};

} // namespace gemfinder
