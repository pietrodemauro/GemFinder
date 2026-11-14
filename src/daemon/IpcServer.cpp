#include "daemon/IpcServer.hpp"
#include "common/Logger.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <poll.h>
#include <vector>
#include <cstring>

namespace gemfinder {

IpcServer::IpcServer(fs::path socket_path, RequestHandler handler)
    : socket_path_(std::move(socket_path)), handler_(std::move(handler)) {}

IpcServer::~IpcServer() {
    stop();
}

bool IpcServer::start() {
    if (running_.load()) return true;

    // Remove any stale socket file
    unlink(socket_path_.c_str());

    server_fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (server_fd_ < 0) {
        Logger::error("IpcServer: creazione socket fallita: " + std::string(std::strerror(errno)));
        return false;
    }

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        Logger::error("IpcServer: bind fallito su " + socket_path_.string() + ": " + std::string(std::strerror(errno)));
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (listen(server_fd_, 16) < 0) {
        Logger::error("IpcServer: listen fallito: " + std::string(std::strerror(errno)));
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    running_.store(true);
    accept_thread_ = std::jthread([this]() { runLoop(); });
    Logger::info("IpcServer in ascolto su socket: " + socket_path_.string());
    return true;
}

void IpcServer::stop() {
    if (!running_.load()) return;

    running_.store(false);
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }

    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }
    unlink(socket_path_.c_str());
    Logger::info("IpcServer arrestato.");
}

void IpcServer::runLoop() {
    while (running_.load()) {
        struct pollfd pfd = {server_fd_, POLLIN, 0};
        int ret = poll(&pfd, 1, 200);

        if (ret > 0 && (pfd.revents & POLLIN)) {
            int client_fd = accept4(server_fd_, nullptr, nullptr, SOCK_CLOEXEC);
            if (client_fd >= 0) {
                handleClient(client_fd);
            }
        }
    }
}

void IpcServer::handleClient(int client_fd) {
    std::vector<char> buf(4096);
    std::string request_str;

    while (true) {
        ssize_t bytes = read(client_fd, buf.data(), buf.size());
        if (bytes <= 0) break;
        request_str.append(buf.data(), static_cast<size_t>(bytes));
        if (request_str.find('\n') != std::string::npos) {
            break;
        }
    }

    IpcRequest req;
    IpcResponse resp;

    if (deserializeRequest(request_str, req)) {
        if (handler_) {
            resp = handler_(req);
        } else {
            resp.success = false;
            resp.error_message = "No handler configured";
        }
    } else {
        resp.success = false;
        resp.error_message = "Malformed request";
    }

    std::string response_str = serializeResponse(resp);
    write(client_fd, response_str.data(), response_str.size());
    close(client_fd);
}

} // namespace gemfinder
