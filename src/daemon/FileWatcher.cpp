#include "daemon/FileWatcher.hpp"
#include "common/Logger.hpp"

#include <sys/inotify.h>
#include <unistd.h>
#include <poll.h>
#include <vector>
#include <cstring>

namespace gemfinder {

FileWatcher::FileWatcher(EventCallback callback, std::chrono::milliseconds debounce_delay)
    : callback_(std::move(callback)), debouncer_(debounce_delay) {}

FileWatcher::~FileWatcher() {
    stop();
}

bool FileWatcher::shouldIgnore(const fs::path& path) const {
    std::string filename = path.filename().string();
    if (!filename.empty() && filename.front() == '.' && filename != "." && filename != "..") {
        return true;
    }
    return excluded_names_.contains(filename);
}

void FileWatcher::addWatchRecursive(const fs::path& root) {
    std::error_code ec;
    if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) return;

    uint32_t mask = IN_CREATE | IN_MODIFY | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_CLOSE_WRITE;
    int wd = inotify_add_watch(inotify_fd_, root.c_str(), mask);
    if (wd >= 0) {
        wd_to_path_[wd] = root;
        path_to_wd_[root.string()] = wd;
    }

    for (const auto& entry : fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied, ec)) {
        if (entry.is_directory(ec)) {
            if (shouldIgnore(entry.path())) {
                continue;
            }
            int sub_wd = inotify_add_watch(inotify_fd_, entry.path().c_str(), mask);
            if (sub_wd >= 0) {
                wd_to_path_[sub_wd] = entry.path();
                path_to_wd_[entry.path().string()] = sub_wd;
            }
        }
    }
}

bool FileWatcher::addWatch(const fs::path& directory_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (inotify_fd_ < 0) return false;
    addWatchRecursive(directory_path);
    return true;
}

void FileWatcher::removeWatch(const fs::path& directory_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (inotify_fd_ < 0) return;

    auto it = path_to_wd_.find(directory_path.string());
    if (it != path_to_wd_.end()) {
        inotify_rm_watch(inotify_fd_, it->second);
        wd_to_path_.erase(it->second);
        path_to_wd_.erase(it);
    }
}

size_t FileWatcher::watchCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return wd_to_path_.size();
}

bool FileWatcher::start() {
    if (running_.load()) return true;

    inotify_fd_ = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    if (inotify_fd_ < 0) {
        Logger::error("FileWatcher: inotify_init1 failed: " + std::string(std::strerror(errno)));
        return false;
    }

    running_.store(true);
    worker_thread_ = std::jthread([this]() { runLoop(); });
    Logger::info("FileWatcher started successfully.");
    return true;
}

void FileWatcher::stop() {
    if (!running_.load()) return;

    running_.store(false);
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (inotify_fd_ >= 0) {
        for (const auto& [wd, _] : wd_to_path_) {
            inotify_rm_watch(inotify_fd_, wd);
        }
        wd_to_path_.clear();
        path_to_wd_.clear();
        close(inotify_fd_);
        inotify_fd_ = -1;
    }

    debouncer_.clear();
    Logger::info("FileWatcher stopped.");
}

void FileWatcher::runLoop() {
    constexpr size_t kBufSize = 4096;
    alignas(struct inotify_event) char buffer[kBufSize];

    while (running_.load()) {
        struct pollfd pfd = {inotify_fd_, POLLIN, 0};
        int ret = poll(&pfd, 1, 200); // 200 ms poll tick

        if (ret > 0 && (pfd.revents & POLLIN)) {
            ssize_t len = read(inotify_fd_, buffer, sizeof(buffer));
            if (len > 0) {
                ssize_t i = 0;
                std::lock_guard<std::mutex> lock(mutex_);

                while (i < len) {
                    auto* event = reinterpret_cast<struct inotify_event*>(&buffer[i]);

                    if (event->len > 0) {
                        auto wd_it = wd_to_path_.find(event->wd);
                        if (wd_it != wd_to_path_.end()) {
                            fs::path full_path = wd_it->second / event->name;

                            if (!shouldIgnore(full_path)) {
                                if (event->mask & IN_ISDIR) {
                                    if (event->mask & (IN_CREATE | IN_MOVED_TO)) {
                                        addWatchRecursive(full_path);
                                    }
                                } else {
                                    if (event->mask & (IN_CREATE | IN_MOVED_TO)) {
                                        debouncer_.addEvent(full_path, FileEvent::Created);
                                    } else if (event->mask & (IN_MODIFY | IN_CLOSE_WRITE)) {
                                        debouncer_.addEvent(full_path, FileEvent::Modified);
                                    } else if (event->mask & (IN_DELETE | IN_MOVED_FROM)) {
                                        debouncer_.addEvent(full_path, FileEvent::Deleted);
                                    }
                                }
                            }
                        }
                    }

                    i += static_cast<ssize_t>(sizeof(struct inotify_event) + event->len);
                }
            }
        }

        // Process debounced events
        auto expired = debouncer_.pollExpired();
        for (const auto& [path, ev] : expired) {
            if (callback_) {
                callback_(path, ev);
            }
        }
    }
}

} // namespace gemfinder
