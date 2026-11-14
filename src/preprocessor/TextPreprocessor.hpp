#pragma once

#include "common/Types.hpp"
#include <string>
#include <vector>
#include <string_view>
#include <filesystem>

namespace gemfinder {

namespace fs = std::filesystem;

struct TextChunk {
    int64_t start_line{0};
    int64_t end_line{0};
    size_t start_char{0};
    size_t end_char{0};
    std::string text;
    std::string preview;
};

class TextPreprocessor {
public:
    struct Config {
        size_t max_chunk_chars{1000};    // ~200-250 words
        size_t overlap_chars{150};       // Overlap between consecutive chunks
        size_t min_chunk_chars{50};      // Discard chunks smaller than this unless entire file
        size_t max_chunks_per_file{100}; // Cap chunks to avoid stalling on huge logs/CSVs
    };

    TextPreprocessor();
    explicit TextPreprocessor(Config config);

    // Process a text file from disk
    [[nodiscard]] std::vector<TextChunk> processFile(const fs::path& path) const;

    // Process a raw text string
    [[nodiscard]] std::vector<TextChunk> processString(std::string_view content) const;

    // Format prompt according to EmbeddingGemma 2 spec:
    // "title: {filename} | text: {chunk_text}"
    [[nodiscard]] static std::string formatDocumentPrompt(const fs::path& filepath, std::string_view chunk_text);

    // Format query prompt according to EmbeddingGemma 2 spec:
    // "task: search result | query: {query_text}"
    [[nodiscard]] static std::string formatQueryPrompt(std::string_view query_text);

private:
    Config config_;
};

} // namespace gemfinder
