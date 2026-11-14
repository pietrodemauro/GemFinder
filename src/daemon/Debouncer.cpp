#include "daemon/Debouncer.hpp"

namespace gemfinder {

void Debouncer::addEvent(const fs::path& path, FileEvent event) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = path.string();

    auto it = pending_.find(key);
    if (it != pending_.end()) {
        // Coalesce events: if already Deleted and new is Created/Modified -> Modified
        if (it->second.event == FileEvent::Deleted && event != FileEvent::Deleted) {
            it->second.event = FileEvent::Modified;
        } else {
            it->second.event = event;
        }
        it->second.last_seen = Clock::now();
    } else {
        pending_[key] = Entry{event, Clock::now()};
    }
}

std::vector<std::pair<fs::path, FileEvent>> Debouncer::pollExpired() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<fs::path, FileEvent>> expired;
    auto now = Clock::now();

    for (auto it = pending_.begin(); it != pending_.end();) {
        if (now - it->second.last_seen >= delay_) {
            expired.emplace_back(it->first, it->second.event);
            it = pending_.erase(it);
        } else {
            ++it;
        }
    }

    return expired;
}

size_t Debouncer::pendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_.size();
}

void Debouncer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.clear();
}

} // namespace gemfinder
