#include "inference/MockEngine.hpp"
#include "inference/Normalization.hpp"
#include "common/Hasher.hpp"
#include <sstream>
#include <cmath>
#include <algorithm>

namespace gemfinder {

DeterministicMockEngine::DeterministicMockEngine(size_t dimension)
    : dimension_(dimension) {}

bool DeterministicMockEngine::loadModel(const fs::path& model_path) {
    (void)model_path;
    return true;
}

Embedding DeterministicMockEngine::embedText(std::string_view text) {
    std::vector<float> raw(dimension_, 0.05f); // base bias

    if (text.empty()) {
        return applyMrlAndNormalize(raw, dimension_);
    }

    // 1. Word token projections
    std::string token;
    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            token += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        } else if (!token.empty()) {
            uint64_t h = Hasher::hashString(token);
            size_t idx1 = h % dimension_;
            size_t idx2 = (h >> 16) % dimension_;
            float sign1 = ((h >> 32) & 1) ? 1.0f : -1.0f;
            float sign2 = ((h >> 33) & 1) ? 1.0f : -1.0f;

            raw[idx1] += sign1 * 1.5f;
            raw[idx2] += sign2 * 1.0f;
            token.clear();
        }
    }
    if (!token.empty()) {
        uint64_t h = Hasher::hashString(token);
        size_t idx1 = h % dimension_;
        raw[idx1] += 1.5f;
    }

    // 2. Character 3-gram projections for subword robustness
    if (text.size() >= 3) {
        for (size_t i = 0; i <= text.size() - 3; ++i) {
            std::string_view trigram = text.substr(i, 3);
            uint64_t h = Hasher::hashString(trigram);
            size_t idx = h % dimension_;
            float sign = ((h >> 8) & 1) ? 0.3f : -0.3f;
            raw[idx] += sign;
        }
    }

    return applyMrlAndNormalize(raw, dimension_);
}

Embedding DeterministicMockEngine::embedImage(std::span<const float> rgb_pixels, int width, int height) {
    std::vector<float> raw(dimension_, 0.0f);

    if (rgb_pixels.empty() || width <= 0 || height <= 0) {
        return applyMrlAndNormalize(raw, dimension_);
    }

    // Compute tile averages and spatial hashes
    constexpr int kTileDim = 8;
    int tile_w = std::max(1, width / kTileDim);
    int tile_h = std::max(1, height / kTileDim);

    for (int ty = 0; ty < kTileDim; ++ty) {
        for (int tx = 0; tx < kTileDim; ++tx) {
            float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
            int count = 0;

            for (int y = ty * tile_h; y < std::min(height, (ty + 1) * tile_h); ++y) {
                for (int x = tx * tile_w; x < std::min(width, (tx + 1) * tile_w); ++x) {
                    size_t idx = static_cast<size_t>((y * width + x) * 3);
                    if (idx + 2 < rgb_pixels.size()) {
                        r_sum += rgb_pixels[idx + 0];
                        g_sum += rgb_pixels[idx + 1];
                        b_sum += rgb_pixels[idx + 2];
                        count++;
                    }
                }
            }

            if (count > 0) {
                float avg_r = r_sum / static_cast<float>(count);
                float avg_g = g_sum / static_cast<float>(count);
                float avg_b = b_sum / static_cast<float>(count);

                size_t bucket = static_cast<size_t>((ty * kTileDim + tx) * 3) % dimension_;
                raw[bucket] += avg_r;
                raw[(bucket + 1) % dimension_] += avg_g;
                raw[(bucket + 2) % dimension_] += avg_b;
            }
        }
    }

    return applyMrlAndNormalize(raw, dimension_);
}

Embedding DeterministicMockEngine::embedAudio(std::span<const float> pcm_samples, uint32_t sample_rate) {
    std::vector<float> raw(dimension_, 0.0f);

    if (pcm_samples.empty() || sample_rate == 0) {
        return applyMrlAndNormalize(raw, dimension_);
    }

    // Compute energy bands and zero-crossing counts
    size_t step = std::max<size_t>(1, pcm_samples.size() / dimension_);
    for (size_t i = 0; i < dimension_; ++i) {
        size_t start = i * step;
        size_t end = std::min(start + step, pcm_samples.size());
        if (start >= pcm_samples.size()) break;

        float energy = 0.0f;
        int zero_crossings = 0;
        for (size_t j = start; j < end; ++j) {
            energy += pcm_samples[j] * pcm_samples[j];
            if (j > start && ((pcm_samples[j] >= 0.0f && pcm_samples[j - 1] < 0.0f) ||
                              (pcm_samples[j] < 0.0f && pcm_samples[j - 1] >= 0.0f))) {
                zero_crossings++;
            }
        }

        raw[i] = std::sqrt(energy / static_cast<float>(std::max<size_t>(1, end - start))) +
                 static_cast<float>(zero_crossings) * 0.01f;
    }

    return applyMrlAndNormalize(raw, dimension_);
}

} // namespace gemfinder
