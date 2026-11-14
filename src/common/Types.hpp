#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>
#include <optional>

namespace gemfinder {

namespace fs = std::filesystem;

using FileId = int64_t;
using ChunkId = int64_t;

// Dimension of the Matryoshka Representation Learning (MRL) embedding
inline constexpr size_t kEmbeddingDim = 256;
inline constexpr size_t kRawEmbeddingDim = 768;

enum class Modality : uint8_t {
    Text,
    Image,
    Audio,
    Video,
    Binary,
    Mixed
};

[[nodiscard]] inline std::string modalityToString(Modality m) {
    switch (m) {
        case Modality::Text:   return "text";
        case Modality::Image:  return "image";
        case Modality::Audio:  return "audio";
        case Modality::Video:  return "video";
        case Modality::Binary: return "binary";
        case Modality::Mixed:  return "mixed";
    }
    return "unknown";
}

[[nodiscard]] inline Modality stringToModality(std::string_view sv) {
    if (sv == "text")   return Modality::Text;
    if (sv == "image")  return Modality::Image;
    if (sv == "audio")  return Modality::Audio;
    if (sv == "video")  return Modality::Video;
    if (sv == "binary") return Modality::Binary;
    return Modality::Mixed;
}

enum class IndexStatus : uint8_t {
    Pending,
    Indexed,
    Error
};

[[nodiscard]] inline std::string statusToString(IndexStatus s) {
    switch (s) {
        case IndexStatus::Pending: return "pending";
        case IndexStatus::Indexed: return "indexed";
        case IndexStatus::Error:   return "error";
    }
    return "unknown";
}

struct FileRecord {
    FileId id{0};
    fs::path filepath;
    std::string filename;
    std::string extension;
    uint64_t file_size{0};
    int64_t mtime{0};
    uint64_t xxhash64{0};
    Modality modality{Modality::Text};
    IndexStatus status{IndexStatus::Pending};
};

struct ChunkRecord {
    ChunkId id{0};
    FileId file_id{0};
    std::string chunk_type;      // "text_paragraph", "image_frame", "audio_segment"
    int64_t start_offset{0};     // Line, second, or page
    int64_t end_offset{0};
    std::string preview_text;
};

using Embedding = std::vector<float>;

struct SearchResult {
    FileId file_id{0};
    ChunkId chunk_id{0};
    std::string filepath;
    std::string filename;
    std::string preview_text;
    std::string chunk_type;
    int64_t start_offset{0};
    double score{0.0};
    int rank_vector{0};
    int rank_lexical{0};
};

} // namespace gemfinder
