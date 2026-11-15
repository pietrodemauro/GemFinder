#pragma once

#include "common/Types.hpp"
#include <filesystem>
#include <string>

namespace gemfinder {

namespace fs = std::filesystem;

inline Modality detectResultModality(const SearchResult& item) {
    if (item.chunk_type == "text_paragraph" || item.chunk_type == "pdf_page") {
        return Modality::Text;
    }
    if (item.chunk_type == "image_frame") {
        return Modality::Image;
    }
    if (item.chunk_type == "audio_segment") {
        return Modality::Audio;
    }

    // Fallback to file extension
    fs::path p(item.filepath);
    std::string ext = p.extension().string();
    for (auto& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" ||
        ext == ".webp" || ext == ".gif" || ext == ".svg") {
        return Modality::Image;
    }
    if (ext == ".wav" || ext == ".mp3" || ext == ".flac" || ext == ".ogg" ||
        ext == ".m4a" || ext == ".aac") {
        return Modality::Audio;
    }
    return Modality::Text;
}

} // namespace gemfinder
