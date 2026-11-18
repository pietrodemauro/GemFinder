#pragma once

#include <filesystem>
#include <string>
#include <cstdlib>

namespace gemfinder {

namespace fs = std::filesystem;

[[nodiscard]] inline fs::path getDefaultDatabasePath() {
    // 1. Environment variable override
    if (const char* env_path = std::getenv("GEMFINDER_DB")) {
        return fs::path(env_path);
    }

    // 2. Standard user directory (~/.local/share/gemfinder/gemfinder_metadata.db)
    const char* home = std::getenv("HOME");
    fs::path standard_dir;
    if (const char* xdg_data = std::getenv("XDG_DATA_HOME")) {
        standard_dir = fs::path(xdg_data) / "gemfinder";
    } else if (home) {
        standard_dir = fs::path(home) / ".local" / "share" / "gemfinder";
    }

    fs::path standard_db = standard_dir / "gemfinder_metadata.db";
    if (!standard_dir.empty() && fs::exists(standard_db)) {
        return standard_db;
    }

    // 3. Local database in current working directory (e.g. during development in repository)
    if (fs::exists("gemfinder_metadata.db")) {
        return fs::absolute("gemfinder_metadata.db");
    }
    if (fs::exists("gemfinder.db")) {
        return fs::absolute("gemfinder.db");
    }

    // 4. Default to standard directory if home is available, creating it
    if (!standard_dir.empty()) {
        std::error_code ec;
        fs::create_directories(standard_dir, ec);
        return standard_db;
    }

    return fs::absolute("gemfinder_metadata.db");
}

[[nodiscard]] inline fs::path getVectorStorePath(const fs::path& db_path) {
    return db_path.parent_path() / (db_path.stem().string() + ".usearch");
}

} // namespace gemfinder
