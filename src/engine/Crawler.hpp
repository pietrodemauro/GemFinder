#pragma once

#include <filesystem>
#include <vector>
#include <string>
#include <unordered_set>
#include <functional>

namespace gemfinder {

namespace fs = std::filesystem;

class Crawler {
public:
    struct Options {
        bool follow_symlinks{false};
        size_t max_file_size_bytes{100 * 1024 * 1024}; // Skip files > 100MB
        std::unordered_set<std::string> excluded_directories{
            ".git", ".svn", ".hg", "node_modules", "build", "cmake-build-debug",
            "cmake-build-release", ".cache", ".idea", ".vscode", "target", "vendor",
            "__pycache__", ".cargo", "dist", "third_party"
        };
    };

    Crawler();
    explicit Crawler(Options options);

    // Crawl directory recursively and collect all eligible file paths
    [[nodiscard]] std::vector<fs::path> crawl(
        const fs::path& root_path,
        const std::function<void(const fs::path&)>& on_file_found = nullptr
    ) const;

    [[nodiscard]] bool isExcluded(const fs::path& path) const;

private:
    Options options_;
};

} // namespace gemfinder
