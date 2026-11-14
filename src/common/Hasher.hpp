#pragma once

#include <cstdint>
#include <string_view>
#include <filesystem>
#include <fstream>
#include <vector>
#include <xxhash.h>

namespace gemfinder {

namespace fs = std::filesystem;

class Hasher {
public:
    // Hash in-memory buffer using XXH64
    static uint64_t hashBuffer(const void* data, size_t length, uint64_t seed = 0) {
        return XXH64(data, length, seed);
    }

    static uint64_t hashString(std::string_view sv, uint64_t seed = 0) {
        return XXH64(sv.data(), sv.size(), seed);
    }

    // Fast streaming file hash using 64KB buffer
    static uint64_t hashFile(const fs::path& path, uint64_t seed = 0) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            return 0;
        }

        XXH64_state_t* state = XXH64_createState();
        if (!state) {
            return 0;
        }

        XXH64_reset(state, seed);

        constexpr size_t kBufferSize = 64 * 1024; // 64 KB
        std::vector<char> buffer(kBufferSize);

        while (file.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) || file.gcount() > 0) {
            XXH64_update(state, buffer.data(), static_cast<size_t>(file.gcount()));
        }

        const uint64_t hash = XXH64_digest(state);
        XXH64_freeState(state);
        return hash;
    }
};

} // namespace gemfinder
