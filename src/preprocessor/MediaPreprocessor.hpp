#pragma once

#include "common/Types.hpp"
#include "preprocessor/TextPreprocessor.hpp"
#include "preprocessor/ImagePreprocessor.hpp"
#include "preprocessor/AudioPreprocessor.hpp"
#include <filesystem>
#include <vector>
#include <optional>

namespace gemfinder {

namespace fs = std::filesystem;

struct PreparedChunk {
    ChunkRecord record;
    std::string prompt;
    std::vector<float> raw_media_buffer; // RGB pixels for image, 16kHz PCM for audio, empty for text
};

struct ProcessedPayload {
    FileRecord file;
    std::vector<PreparedChunk> chunks;
};

class MediaPreprocessor {
public:
    MediaPreprocessor();

    // Detect modality from file path/extension
    [[nodiscard]] static Modality detectModality(const fs::path& path);

    // Full processing of a single file
    [[nodiscard]] std::optional<ProcessedPayload> processFile(const fs::path& path) const;

    // Direct access to sub-preprocessors
    [[nodiscard]] const TextPreprocessor& text() const noexcept { return text_; }
    [[nodiscard]] const ImagePreprocessor& image() const noexcept { return image_; }
    [[nodiscard]] const AudioPreprocessor& audio() const noexcept { return audio_; }

private:
    TextPreprocessor text_;
    ImagePreprocessor image_;
    AudioPreprocessor audio_;
};

} // namespace gemfinder
