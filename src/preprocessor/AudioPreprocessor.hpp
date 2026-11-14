#pragma once

#include "common/Types.hpp"
#include <string>
#include <vector>
#include <filesystem>
#include <optional>

namespace gemfinder {

namespace fs = std::filesystem;

inline constexpr uint32_t kTargetSampleRate = 16000; // 16 kHz mono for EmbeddingGemma 2

struct AudioChunk {
    int64_t start_second{0};
    int64_t end_second{0};
    size_t sample_offset{0};
    size_t sample_count{0};
    std::vector<float> pcm_samples; // 16 kHz mono float32
    std::string preview_text;
    std::string document_prompt;
};

struct ProcessedAudio {
    uint32_t sample_rate{kTargetSampleRate};
    uint32_t channels{1};
    double total_duration_seconds{0.0};
    std::vector<AudioChunk> chunks;
};

class AudioPreprocessor {
public:
    struct Config {
        uint32_t target_sample_rate{kTargetSampleRate};
        double window_duration_seconds{30.0}; // 30 second window
        double overlap_duration_seconds{5.0};  // 5 second overlap
    };

    AudioPreprocessor();
    explicit AudioPreprocessor(Config config);

    // Process audio file (WAV, MP3, FLAC, etc.) from disk
    [[nodiscard]] std::optional<ProcessedAudio> processFile(const fs::path& path) const;

    // Process in-memory raw PCM or audio buffer
    [[nodiscard]] std::optional<ProcessedAudio> processMemory(const void* data, size_t length, const std::string& filename = "") const;

private:
    Config config_;
};

} // namespace gemfinder
