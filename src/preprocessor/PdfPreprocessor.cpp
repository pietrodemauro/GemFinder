#include "preprocessor/PdfPreprocessor.hpp"
#include "common/Logger.hpp"

#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>
#include <poppler/cpp/poppler-page-renderer.h>
#include <poppler/cpp/poppler-image.h>
#include <tesseract/baseapi.h>

#include <algorithm>
#include <memory>
#include <cctype>

namespace gemfinder {

namespace {

std::string sanitizeAndTrim(const std::string& input) {
    if (input.empty()) return {};

    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        if (c == '\0' || c == '\f' || c == '\r') {
            continue;
        }
        out.push_back(c);
    }

    size_t start = 0;
    while (start < out.size() && std::isspace(static_cast<unsigned char>(out[start]))) {
        ++start;
    }
    size_t end = out.size();
    while (end > start && std::isspace(static_cast<unsigned char>(out[end - 1]))) {
        --end;
    }
    return out.substr(start, end - start);
}

size_t countMeaningfulChars(const std::string& s) {
    size_t count = 0;
    for (char c : s) {
        if (!std::isspace(static_cast<unsigned char>(c)) && !std::iscntrl(static_cast<unsigned char>(c))) {
            ++count;
        }
    }
    return count;
}

std::string makePreview(const std::string& text) {
    std::string preview;
    preview.reserve(std::min(text.size(), size_t{160}));

    bool last_was_space = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!last_was_space && !preview.empty()) {
                preview.push_back(' ');
                last_was_space = true;
            }
        } else {
            preview.push_back(c);
            last_was_space = false;
        }
        if (preview.size() >= 160) break;
    }

    if (preview.size() > 157) {
        preview = preview.substr(0, 157) + "...";
    }
    return preview;
}

} // namespace

PdfPreprocessor::PdfPreprocessor() = default;

PdfPreprocessor::PdfPreprocessor(Config config)
    : config_(std::move(config)) {}

PdfPreprocessor::~PdfPreprocessor() = default;
PdfPreprocessor::PdfPreprocessor(PdfPreprocessor&&) noexcept = default;
PdfPreprocessor& PdfPreprocessor::operator=(PdfPreprocessor&&) noexcept = default;

std::vector<PdfChunk> PdfPreprocessor::processFile(const fs::path& path) const {
    std::error_code ec;
    if (!fs::exists(path, ec) || !fs::is_regular_file(path, ec)) {
        return {};
    }

    std::unique_ptr<poppler::document> doc(poppler::document::load_from_file(path.string()));
    if (!doc) {
        Logger::warn("PdfPreprocessor: Unable to open PDF file: " + path.string());
        return {};
    }

    if (doc->is_locked()) {
        Logger::warn("PdfPreprocessor: Password-protected PDF file, skipping: " + path.string());
        return {};
    }

    int total_pages = doc->pages();
    if (total_pages <= 0) {
        return {};
    }

    int pages_to_process = std::min(total_pages, static_cast<int>(config_.max_pages));
    std::vector<PdfChunk> result_chunks;

    // Tesseract engine initialized lazily if OCR is needed
    std::unique_ptr<tesseract::TessBaseAPI> tess;
    bool tess_attempted = false;

    auto get_tesseract = [&]() -> tesseract::TessBaseAPI* {
        if (tess_attempted) return tess.get();
        tess_attempted = true;

        auto api = std::make_unique<tesseract::TessBaseAPI>();

        std::vector<std::string> search_paths = {
            config_.tessdata_path,
            "models/tessdata",
            "../models/tessdata",
            "/usr/share/tessdata"
        };
        std::vector<std::string> lang_candidates = {
            config_.languages,
            "ita+eng",
            "eng",
            "osd"
        };

        bool ok = false;
        for (const auto& sp : search_paths) {
            std::error_code p_ec;
            if (!sp.empty() && !fs::exists(sp, p_ec)) continue;

            for (const auto& lang : lang_candidates) {
                if (api->Init(sp.c_str(), lang.c_str()) == 0) {
                    Logger::info("PdfPreprocessor: Tesseract OCR initialized (" + sp + ", language: " + lang + ")");
                    ok = true;
                    break;
                }
            }
            if (ok) break;
        }

        if (!ok) {
            if (api->Init(nullptr, "eng") == 0 || api->Init(nullptr, nullptr) == 0) {
                Logger::info("PdfPreprocessor: Tesseract OCR initialized with system model");
                ok = true;
            }
        }

        if (ok) {
            tess = std::move(api);
            return tess.get();
        }

        Logger::warn("PdfPreprocessor: Tesseract initialization failed. OCR unavailable.");
        return nullptr;
    };

    for (int p_idx = 0; p_idx < pages_to_process; ++p_idx) {
        std::unique_ptr<poppler::page> page(doc->create_page(p_idx));
        if (!page) continue;

        int page_num = p_idx + 1;
        bool is_ocr = false;
        std::string extracted_text;

        // 1. Try native text extraction
        auto ustr = page->text();
        auto utf8_bytes = ustr.to_utf8();
        std::string native_raw(utf8_bytes.data(), utf8_bytes.size());
        std::string native_clean = sanitizeAndTrim(native_raw);

        if (countMeaningfulChars(native_clean) >= config_.min_native_chars) {
            extracted_text = std::move(native_clean);
        } else {
            // 2. Fallback to OCR via Poppler page renderer + Tesseract
            auto* ocr_api = get_tesseract();
            if (ocr_api) {
                poppler::page_renderer renderer;
                renderer.set_image_format(poppler::image::format_rgb24);
                poppler::image img = renderer.render_page(page.get(), config_.ocr_dpi, config_.ocr_dpi);

                if (img.is_valid()) {
                    ocr_api->SetImage(reinterpret_cast<const unsigned char*>(img.const_data()),
                                      img.width(), img.height(), 3, img.bytes_per_row());
                    char* ocr_text_ptr = ocr_api->GetUTF8Text();
                    if (ocr_text_ptr) {
                        std::string ocr_raw(ocr_text_ptr);
                        delete[] ocr_text_ptr;
                        std::string ocr_clean = sanitizeAndTrim(ocr_raw);
                        if (countMeaningfulChars(ocr_clean) > 0) {
                            extracted_text = std::move(ocr_clean);
                            is_ocr = true;
                        }
                    }
                    ocr_api->Clear();
                }
            } else if (!native_clean.empty()) {
                // Keep minimal native text if OCR is unavailable
                extracted_text = std::move(native_clean);
            }
        }

        if (extracted_text.empty()) {
            continue;
        }

        // 3. Chunking if text is longer than max_chunk_chars
        if (extracted_text.size() <= config_.max_chunk_chars) {
            PdfChunk chunk;
            chunk.page_number = page_num;
            chunk.preview = makePreview(extracted_text);
            chunk.text = std::move(extracted_text);
            chunk.is_ocr = is_ocr;
            result_chunks.push_back(std::move(chunk));
        } else {
            size_t start = 0;
            const size_t step = (config_.max_chunk_chars > config_.overlap_chars)
                                    ? (config_.max_chunk_chars - config_.overlap_chars)
                                    : config_.max_chunk_chars;

            while (start < extracted_text.size()) {
                size_t len = std::min(config_.max_chunk_chars, extracted_text.size() - start);
                std::string slice = sanitizeAndTrim(extracted_text.substr(start, len));

                if (countMeaningfulChars(slice) > 0) {
                    PdfChunk chunk;
                    chunk.page_number = page_num;
                    chunk.preview = makePreview(slice);
                    chunk.text = std::move(slice);
                    chunk.is_ocr = is_ocr;
                    result_chunks.push_back(std::move(chunk));
                }

                if (start + len >= extracted_text.size()) {
                    break;
                }
                start += step;
            }
        }
    }

    if (tess) {
        tess->End();
    }

    return result_chunks;
}

} // namespace gemfinder
