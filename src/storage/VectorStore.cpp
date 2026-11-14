#include "storage/VectorStore.hpp"
#include "common/Logger.hpp"
#include <usearch/index_dense.hpp>
#include <algorithm>
#include <mutex>

namespace gemfinder {

using namespace unum::usearch;

class VectorStore::Impl {
public:
    explicit Impl(size_t dimensions)
        : dimensions_(dimensions),
          metric_(dimensions, metric_kind_t::cos_k, scalar_kind_t::f32_k),
          index_(index_dense_t::make(metric_, index_dense_config_t{})) {
        index_.reserve(1024);
    }

    bool add(ChunkId id, std::span<const float> vector) {
        if (vector.size() != dimensions_) {
            Logger::error("VectorStore dimension mismatch: expected " +
                          std::to_string(dimensions_) + ", got " + std::to_string(vector.size()));
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (index_.size() >= index_.capacity()) {
            index_.reserve(index_.size() * 2 + 128);
        }

        auto res = index_.add(static_cast<index_dense_t::vector_key_t>(id), vector.data());
        if (!res) {
            Logger::error("Failed to add vector to USearch index: " + std::string(res.error.what()));
            return false;
        }
        return true;
    }

    bool remove(ChunkId id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto res = index_.remove(static_cast<index_dense_t::vector_key_t>(id));
        return bool(res);
    }

    std::vector<VectorMatch> search(std::span<const float> query, size_t limit) const {
        std::vector<VectorMatch> matches;
        if (query.size() != dimensions_ || index_.size() == 0 || limit == 0) {
            return matches;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto result = index_.search(query.data(), limit);
        if (!result) {
            return matches;
        }

        std::vector<index_dense_t::vector_key_t> keys(limit);
        std::vector<index_dense_t::distance_t> distances(limit);
        size_t count = result.dump_to(keys.data(), distances.data(), limit);

        matches.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            VectorMatch m;
            m.chunk_id = static_cast<ChunkId>(keys[i]);
            m.distance = distances[i];
            m.similarity = std::clamp(1.0f - distances[i], -1.0f, 1.0f);
            matches.push_back(m);
        }

        return matches;
    }

    size_t size() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_.size();
    }

    bool contains(ChunkId id) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_.contains(static_cast<index_dense_t::vector_key_t>(id));
    }

    bool save(const fs::path& path) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto parent = path.parent_path(); !parent.empty()) {
            fs::create_directories(parent);
        }
        auto res = index_.save(path.c_str());
        if (!res) {
            Logger::error("Failed to save USearch index: " + std::string(res.error.what()));
            return false;
        }
        return true;
    }

    bool load(const fs::path& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        index_ = index_dense_t::make(metric_, index_dense_config_t{});
        auto res = index_.load(path.c_str());
        if (!res) {
            Logger::error("Failed to load USearch index: " + std::string(res.error.what()));
            return false;
        }
        return true;
    }

    bool view(const fs::path& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        index_ = index_dense_t::make(metric_, index_dense_config_t{});
        auto res = index_.view(path.c_str());
        if (!res) {
            Logger::error("Failed to view (mmap) USearch index: " + std::string(res.error.what()));
            return false;
        }
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        index_ = index_dense_t::make(metric_, index_dense_config_t{});
    }

private:
    size_t dimensions_;
    metric_punned_t metric_;
    index_dense_t index_;
    mutable std::mutex mutex_;
};

VectorStore::VectorStore(size_t dimensions)
    : impl_(std::make_unique<Impl>(dimensions)), dimensions_(dimensions) {}

VectorStore::~VectorStore() = default;

VectorStore::VectorStore(VectorStore&&) noexcept = default;
VectorStore& VectorStore::operator=(VectorStore&&) noexcept = default;

bool VectorStore::add(ChunkId id, std::span<const float> vector) {
    return impl_->add(id, vector);
}

bool VectorStore::remove(ChunkId id) {
    return impl_->remove(id);
}

std::vector<VectorMatch> VectorStore::search(std::span<const float> query, size_t limit) const {
    return impl_->search(query, limit);
}

size_t VectorStore::size() const noexcept {
    return impl_->size();
}

bool VectorStore::contains(ChunkId id) const noexcept {
    return impl_->contains(id);
}

bool VectorStore::save(const fs::path& path) const {
    return impl_->save(path);
}

bool VectorStore::load(const fs::path& path) {
    return impl_->load(path);
}

bool VectorStore::view(const fs::path& path) {
    return impl_->view(path);
}

void VectorStore::clear() {
    impl_->clear();
}

} // namespace gemfinder
