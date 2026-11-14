#pragma once

#include "common/Types.hpp"
#include <vector>
#include <span>
#include <cmath>
#include <algorithm>

namespace gemfinder {

// Matryoshka Representation Learning (MRL) truncation & L2 Euclidean normalization
[[nodiscard]] inline Embedding applyMrlAndNormalize(
    std::span<const float> raw_vector,
    size_t target_dim = kEmbeddingDim
) {
    Embedding truncated(target_dim, 0.0f);
    size_t count = std::min(raw_vector.size(), target_dim);
    for (size_t i = 0; i < count; ++i) {
        truncated[i] = raw_vector[i];
    }

    // Compute L2 Euclidean norm: sqrt(sum(v_i^2))
    float sum_sq = 0.0f;
    for (float v : truncated) {
        sum_sq += v * v;
    }

    float norm = std::sqrt(sum_sq);
    if (norm > 1e-12f) {
        float inv_norm = 1.0f / norm;
        for (float& v : truncated) {
            v *= inv_norm;
        }
    }

    return truncated;
}

// Compute cosine similarity between two unit vectors (dot product)
[[nodiscard]] inline float cosineSimilarity(
    std::span<const float> a,
    std::span<const float> b
) {
    if (a.size() != b.size() || a.empty()) return 0.0f;

    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

} // namespace gemfinder
