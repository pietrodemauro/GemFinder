#pragma once

#include "common/Types.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace gemfinder {

namespace fs = std::filesystem;

struct PdfChunk {
    int64_t page_number{1}; // 1-indexed page
    std::string text;
    std::string preview;
    bool is_ocr{false};
};

class PdfPreprocessor {
public:
    struct Config {
        size_t max_pages{100};
        double ocr_dpi{150.0};
        std::string tessdata_path{"models/tessdata"};
        std::string languages{"ita+eng"};
        size_t min_native_chars{20};
        size_t max_chunk_chars{1000};
        size_t overlap_chars{150};
    };

    PdfPreprocessor();
    explicit PdfPreprocessor(Config config);
    ~PdfPreprocessor();

    PdfPreprocessor(PdfPreprocessor&&) noexcept;
    PdfPreprocessor& operator=(PdfPreprocessor&&) noexcept;
    PdfPreprocessor(const PdfPreprocessor&) = delete;
    PdfPreprocessor& operator=(const PdfPreprocessor&) = delete;

    [[nodiscard]] std::vector<PdfChunk> processFile(const fs::path& path) const;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

private:
    Config config_;
};

} // namespace gemfinder
