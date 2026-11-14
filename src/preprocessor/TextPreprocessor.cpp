#include "preprocessor/TextPreprocessor.hpp"
#include "common/Logger.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace gemfinder {

TextPreprocessor::TextPreprocessor() : config_{} {}
TextPreprocessor::TextPreprocessor(Config config) : config_(config) {}

std::vector<TextChunk> TextPreprocessor::processFile(const fs::path& path) const {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        Logger::error("TextPreprocessor: cannot open file " + path.string());
        return {};
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return processString(ss.str());
}

std::vector<TextChunk> TextPreprocessor::processString(std::string_view content) const {
    std::vector<TextChunk> chunks;
    if (content.empty()) {
        return chunks;
    }

    // Small content: return single chunk
    if (content.size() <= config_.max_chunk_chars) {
        TextChunk chunk;
        chunk.start_char = 0;
        chunk.end_char = content.size();
        chunk.start_line = 1;
        chunk.end_line = static_cast<int64_t>(std::count(content.begin(), content.end(), '\n') + 1);
        chunk.text = std::string(content);
        // Preview: first 120 chars
        chunk.preview = chunk.text.substr(0, std::min<size_t>(120, chunk.text.size()));
        chunks.push_back(std::move(chunk));
        return chunks;
    }

    // Sliding window with semantic paragraph / sentence boundary snapping
    size_t cursor = 0;
    int64_t current_line = 1;

    while (cursor < content.size()) {
        size_t target_end = std::min(cursor + config_.max_chunk_chars, content.size());
        size_t split_point = target_end;

        if (target_end < content.size()) {
            // 1. Look for paragraph break (\n\n) in the last 25% of the window
            size_t search_start = cursor + (config_.max_chunk_chars * 3) / 4;
            auto para_pos = content.substr(search_start, target_end - search_start).rfind("\n\n");
            if (para_pos != std::string_view::npos) {
                split_point = search_start + para_pos + 2;
            } else {
                // 2. Look for sentence end (. / ! / ?)
                auto dot_pos = content.substr(search_start, target_end - search_start).rfind(". ");
                if (dot_pos != std::string_view::npos) {
                    split_point = search_start + dot_pos + 2;
                } else {
                    // 3. Look for newline or space
                    auto space_pos = content.substr(search_start, target_end - search_start).rfind(' ');
                    if (space_pos != std::string_view::npos) {
                        split_point = search_start + space_pos + 1;
                    }
                }
            }
        }

        std::string_view chunk_view = content.substr(cursor, split_point - cursor);

        // Count lines in this slice
        int64_t chunk_lines = static_cast<int64_t>(std::count(chunk_view.begin(), chunk_view.end(), '\n'));

        TextChunk chunk;
        chunk.start_char = cursor;
        chunk.end_char = split_point;
        chunk.start_line = current_line;
        chunk.end_line = current_line + chunk_lines;
        chunk.text = std::string(chunk_view);

        // Preview snippet: clean up leading/trailing whitespaces and take ~120 chars
        size_t first_non_ws = chunk.text.find_first_not_of(" \t\r\n");
        if (first_non_ws != std::string::npos) {
            std::string preview_sub = chunk.text.substr(first_non_ws, 120);
            std::replace(preview_sub.begin(), preview_sub.end(), '\n', ' ');
            chunk.preview = preview_sub;
        } else {
            chunk.preview = chunk.text.substr(0, std::min<size_t>(120, chunk.text.size()));
        }

        if (chunk.text.size() >= config_.min_chunk_chars || chunks.empty()) {
            chunks.push_back(std::move(chunk));
        }

        if (chunks.size() >= config_.max_chunks_per_file) {
            break;
        }

        // Advance cursor with overlap
        if (split_point >= content.size()) {
            break;
        }

        size_t step = (split_point > cursor + config_.overlap_chars)
                          ? (split_point - config_.overlap_chars)
                          : split_point;

        // Advance line counter for the step
        std::string_view stepped_view = content.substr(cursor, step - cursor);
        current_line += std::count(stepped_view.begin(), stepped_view.end(), '\n');
        cursor = step;
    }

    return chunks;
}

std::string TextPreprocessor::formatDocumentPrompt(const fs::path& filepath, std::string_view chunk_text) {
    std::string prompt;
    prompt.reserve(filepath.string().size() + chunk_text.size() + 20);
    prompt.append("title: ");
    prompt.append(filepath.filename().string());
    prompt.append(" | text: ");
    prompt.append(chunk_text);
    return prompt;
}

std::string TextPreprocessor::formatQueryPrompt(std::string_view query_text) {
    std::string prompt;
    prompt.reserve(query_text.size() + 32);
    prompt.append("task: search result | query: ");
    prompt.append(query_text);
    return prompt;
}

} // namespace gemfinder
