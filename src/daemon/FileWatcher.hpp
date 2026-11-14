#pragma once

#include "daemon/Debouncer.hpp"
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>

namespace gemfinder {

namespace fs = std::filesystem;

class FileWatcher {
public:
    using EventCallback = std::function<void(const fs::path& path, FileEvent event)>;

    explicit FileWatcher(
        EventCallback callback,
        std::chrono::milliseconds debounce_delay = std::chrono::milliseconds(2000)
    );
    ~FileWatcher();

    // Start / Stop watching
    bool start();
    void stop();

    // Add / Remove directory to watch
    bool addWatch(const fs::path& directory_path);
    void removeWatch(const fs::path& directory_path);

    [[nodiscard]] bool isRunning() const noexcept { return running_.load(); }
    [[nodiscard]] size_t watchCount() const;

private:
    EventCallback callback_;
    Debouncer debouncer_;
    int inotify_fd_{-1};
    std::atomic<bool> running_{false};
    std::jthread worker_thread_;

    std::unordered_map<int, fs::path> wd_to_path_;
    std::unordered_map<std::string, int> path_to_wd_;
    std::unordered_set<std::string> excluded_names_{
        ".git", ".svn", ".hg", "node_modules", "build", "cmake-build-debug",
        "cmake-build-release", ".cache", ".idea", ".vscode", "target", "vendor",
        "__pycache__", ".cargo", "dist", "third_party"
    };

    mutable std::mutex mutex_;

    void runLoop();
    void addWatchRecursive(const fs::path& root);
    bool shouldIgnore(const fs::path& path) const;
};

} // namespace gemfinder
