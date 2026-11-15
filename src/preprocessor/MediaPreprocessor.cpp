#include "preprocessor/MediaPreprocessor.hpp"
#include "common/Hasher.hpp"
#include "common/Logger.hpp"
#include <algorithm>
#include <chrono>

namespace gemfinder {

MediaPreprocessor::MediaPreprocessor() = default;

Modality MediaPreprocessor::detectModality(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    // Image extensions
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
        ext == ".webp" || ext == ".tga" || ext == ".gif" || ext == ".hdr" || ext == ".pic") {
        return Modality::Image;
    }

    // Audio extensions
    if (ext == ".wav" || ext == ".mp3" || ext == ".flac" || ext == ".ogg" ||
        ext == ".m4a" || ext == ".aac" || ext == ".opus" || ext == ".wma") {
        return Modality::Audio;
    }

    // Video extensions
    if (ext == ".mp4" || ext == ".mkv" || ext == ".avi" || ext == ".mov" ||
        ext == ".webm" || ext == ".flv") {
        return Modality::Video;
    }

    // Common Text/Source code extensions
    if (ext == ".txt" || ext == ".md" || ext == ".markdown" || ext == ".c" ||
        ext == ".cpp" || ext == ".h" || ext == ".hpp" || ext == ".cc" ||
        ext == ".cxx" || ext == ".py" || ext == ".js" || ext == ".ts" ||
        ext == ".tsx" || ext == ".jsx" || ext == ".json" || ext == ".yaml" ||
        ext == ".yml" || ext == ".toml" || ext == ".ini" || ext == ".cfg" ||
        ext == ".sh" || ext == ".bash" || ext == ".zsh" || ext == ".html" ||
        ext == ".htm" || ext == ".css" || ext == ".scss" || ext == ".xml" ||
        ext == ".csv" || ext == ".tsv" || ext == ".sql" || ext == ".rs" ||
        ext == ".go" || ext == ".java" || ext == ".kt" || ext == ".lua" ||
        ext == ".php" || ext == ".rb" || ext == ".swift" || ext == ".tex") {
        return Modality::Text;
    }

    // PDF documents
    if (ext == ".pdf") {
        return Modality::Text;
    }

    // If extension is empty or unknown binary
    return Modality::Binary;
}

std::optional<ProcessedPayload> MediaPreprocessor::processFile(const fs::path& path) const {
    std::error_code ec;
    if (!fs::exists(path, ec) || !fs::is_regular_file(path, ec)) {
        return std::nullopt;
    }

    uint64_t file_size = fs::file_size(path, ec);
    if (ec) file_size = 0;

    int64_t mtime = getFileMtime(path);
    uint64_t hash = Hasher::hashFile(path);

    Modality mod = detectModality(path);

    ProcessedPayload payload;
    payload.file.filepath = path;
    payload.file.filename = path.filename().string();
    payload.file.extension = path.extension().string();
    payload.file.file_size = file_size;
    payload.file.mtime = mtime;
    payload.file.xxhash64 = hash;
    payload.file.modality = mod;
    payload.file.status = IndexStatus::Pending;

    switch (mod) {
        case Modality::Text: {
            std::string ext_lower = path.extension().string();
            std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });

            if (ext_lower == ".pdf") {
                auto pdf_chunks = pdf_.processFile(path);
                payload.chunks.reserve(pdf_chunks.size());
                for (const auto& pc : pdf_chunks) {
                    PreparedChunk chunk;
                    chunk.record.chunk_type = "pdf_page";
                    chunk.record.start_offset = pc.page_number;
                    chunk.record.end_offset = pc.page_number;
                    chunk.record.preview_text = pc.preview;
                    chunk.prompt = TextPreprocessor::formatDocumentPrompt(path, pc.text);
                    payload.chunks.push_back(std::move(chunk));
                }

                if (payload.chunks.empty()) {
                    // Fallback to filename/metadata if PDF produced no text
                    PreparedChunk chunk;
                    chunk.record.chunk_type = "file_metadata";
                    chunk.record.start_offset = 0;
                    chunk.record.end_offset = 0;
                    chunk.record.preview_text = path.filename().string() + " (" + std::to_string(file_size) + " bytes)";
                    chunk.prompt = "title: " + path.filename().string();
                    payload.chunks.push_back(std::move(chunk));
                }
            } else {
                auto text_chunks = text_.processFile(path);
                payload.chunks.reserve(text_chunks.size());
                for (const auto& tc : text_chunks) {
                    PreparedChunk pc;
                    pc.record.chunk_type = "text_paragraph";
                    pc.record.start_offset = tc.start_line;
                    pc.record.end_offset = tc.end_line;
                    pc.record.preview_text = tc.preview;
                    pc.prompt = TextPreprocessor::formatDocumentPrompt(path, tc.text);
                    payload.chunks.push_back(std::move(pc));
                }
            }
            break;
        }

        case Modality::Image: {
            auto processed_img = image_.processFile(path);
            if (processed_img) {
                PreparedChunk pc;
                pc.record.chunk_type = "image_frame";
                pc.record.start_offset = 0;
                pc.record.end_offset = 0;
                pc.record.preview_text = processed_img->preview_text;
                pc.prompt = processed_img->document_prompt;
                pc.raw_media_buffer = std::move(processed_img->rgb_planar_or_chw);
                payload.chunks.push_back(std::move(pc));
            }
            break;
        }

        case Modality::Audio: {
            auto processed_aud = audio_.processFile(path);
            if (processed_aud) {
                payload.chunks.reserve(processed_aud->chunks.size());
                for (auto& ac : processed_aud->chunks) {
                    PreparedChunk pc;
                    pc.record.chunk_type = "audio_segment";
                    pc.record.start_offset = ac.start_second;
                    pc.record.end_offset = ac.end_second;
                    pc.record.preview_text = ac.preview_text;
                    pc.prompt = ac.document_prompt;
                    pc.raw_media_buffer = std::move(ac.pcm_samples);
                    payload.chunks.push_back(std::move(pc));
                }
            }
            break;
        }

        case Modality::Video:
        case Modality::Binary:
        case Modality::Mixed: {
            // Index file path & metadata only
            PreparedChunk pc;
            pc.record.chunk_type = "file_metadata";
            pc.record.start_offset = 0;
            pc.record.end_offset = 0;
            pc.record.preview_text = path.filename().string() + " (" + std::to_string(file_size) + " bytes)";
            pc.prompt = "title: " + path.filename().string();
            payload.chunks.push_back(std::move(pc));
            break;
        }
    }

    return payload;
}

} // namespace gemfinder
