#include "daemon/IpcClient.hpp"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <poll.h>
#include <vector>
#include <cstring>

namespace gemfinder {

IpcClient::IpcClient(fs::path socket_path) : socket_path_(std::move(socket_path)) {}

std::optional<IpcResponse> IpcClient::send(const IpcRequest& request, int timeout_ms) const {
    int sock_fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sock_fd < 0) return std::nullopt;

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(sock_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(sock_fd);
        return std::nullopt;
    }

    std::string payload = serializeRequest(request);
    if (write(sock_fd, payload.data(), payload.size()) <= 0) {
        close(sock_fd);
        return std::nullopt;
    }

    struct pollfd pfd = {sock_fd, POLLIN, 0};
    int ret = poll(&pfd, 1, timeout_ms);
    if (ret <= 0 || !(pfd.revents & POLLIN)) {
        close(sock_fd);
        return std::nullopt;
    }

    std::string response_str;
    std::vector<char> buf(4096);
    while (true) {
        ssize_t bytes = read(sock_fd, buf.data(), buf.size());
        if (bytes <= 0) break;
        response_str.append(buf.data(), static_cast<size_t>(bytes));
    }
    close(sock_fd);

    IpcResponse response;
    if (deserializeResponse(response_str, response)) {
        return response;
    }
    return std::nullopt;
}

bool IpcClient::isDaemonRunning() const {
    IpcRequest req;
    req.action = IpcAction::Ping;
    auto resp = send(req, 500);
    return resp.has_value() && resp->success;
}

} // namespace gemfinder
