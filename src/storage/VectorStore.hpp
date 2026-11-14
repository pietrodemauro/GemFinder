#pragma once

#include "common/Types.hpp"
#include <vector>
#include <span>
#include <filesystem>
#include <memory>
#include <utility>

namespace gemfinder {

namespace fs = std::filesystem;

struct VectorMatch {
    ChunkId chunk_id{0};
    float distance{0.0f};    // Cosine distance (0 = identical, 2 = opposite)
    float similarity{0.0f};  // Cosine similarity (1 - distance)
};

class VectorStore {
public:
    explicit VectorStore(size_t dimensions = kEmbeddingDim);
    ~VectorStore();

    // Movable, non-copyable
    VectorStore(const VectorStore&) = delete;
    VectorStore& operator=(const VectorStore&) = delete;
    VectorStore(VectorStore&&) noexcept;
    VectorStore& operator=(VectorStore&&) noexcept;

    // Core index operations
    bool add(ChunkId id, std::span<const float> vector);
    bool remove(ChunkId id);
    [[nodiscard]] std::vector<VectorMatch> search(std::span<const float> query, size_t limit = 50) const;

    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] bool contains(ChunkId id) const noexcept;
    [[nodiscard]] size_t dimensions() const noexcept { return dimensions_; }

    // Persistence & Memory mapping
    bool save(const fs::path& path) const;
    bool load(const fs::path& path);
    bool view(const fs::path& path); // Zero-latency memory-mapped view

    void clear();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    size_t dimensions_{kEmbeddingDim};
};

} // namespace gemfinder
