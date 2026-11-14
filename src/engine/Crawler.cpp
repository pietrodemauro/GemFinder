#include "engine/Crawler.hpp"
#include "common/Logger.hpp"
#include <algorithm>

namespace gemfinder {

Crawler::Crawler() : options_{} {}
Crawler::Crawler(Options options) : options_(std::move(options)) {}

bool Crawler::isExcluded(const fs::path& path) const {
    std::string filename = path.filename().string();

    // Exclude hidden files / directories (starting with '.')
    if (!filename.empty() && filename.front() == '.' && filename != "." && filename != "..") {
        return true;
    }

    if (options_.excluded_directories.contains(filename)) {
        return true;
    }

    return false;
}

std::vector<fs::path> Crawler::crawl(
    const fs::path& root_path,
    const std::function<void(const fs::path&)>& on_file_found
) const {
    std::vector<fs::path> results;
    std::error_code ec;

    if (!fs::exists(root_path, ec) || !fs::is_directory(root_path, ec)) {
        Logger::error("Crawler: path non valido o non è una cartella: " + root_path.string());
        return results;
    }

    auto dir_options = fs::directory_options::skip_permission_denied;
    if (options_.follow_symlinks) {
        dir_options |= fs::directory_options::follow_directory_symlink;
    }

    fs::recursive_directory_iterator it(root_path, dir_options, ec);
    fs::recursive_directory_iterator end;

    while (it != end && !ec) {
        const auto& entry = *it;
        const auto& current_path = entry.path();

        // Check if directory should be skipped
        if (entry.is_directory(ec)) {
            if (isExcluded(current_path)) {
                it.disable_recursion_pending();
            }
            it.increment(ec);
            continue;
        }

        // Regular file check
        if (entry.is_regular_file(ec)) {
            if (!isExcluded(current_path)) {
                uintmax_t size = entry.file_size(ec);
                if (!ec && size <= options_.max_file_size_bytes && size > 0) {
                    results.push_back(current_path);
                    if (on_file_found) {
                        on_file_found(current_path);
                    }
                }
            }
        }

        it.increment(ec);
    }

    return results;
}

} // namespace gemfinder
