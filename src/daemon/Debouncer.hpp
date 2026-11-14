#pragma once

#include <filesystem>
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <functional>
#include <vector>

namespace gemfinder {

namespace fs = std::filesystem;

enum class FileEvent {
    Created,
    Modified,
    Deleted
};

class Debouncer {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    explicit Debouncer(std::chrono::milliseconds delay = std::chrono::milliseconds(2000))
        : delay_(delay) {}

    // Enqueue an event for a path
    void addEvent(const fs::path& path, FileEvent event);

    // Collect and remove all events that have been idle for >= delay
    std::vector<std::pair<fs::path, FileEvent>> pollExpired();

    // Check how many pending paths are currently debouncing
    [[nodiscard]] size_t pendingCount() const;

    void clear();

private:
    struct Entry {
        FileEvent event;
        TimePoint last_seen;
    };

    std::chrono::milliseconds delay_;
    std::unordered_map<std::string, Entry> pending_;
    mutable std::mutex mutex_;
};

} // namespace gemfinder
