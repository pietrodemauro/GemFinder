#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include "preprocessor/TextPreprocessor.hpp"
#include "preprocessor/PdfPreprocessor.hpp"
#include "preprocessor/ImagePreprocessor.hpp"
#include "preprocessor/AudioPreprocessor.hpp"
#include "preprocessor/MediaPreprocessor.hpp"
#include "common/Logger.hpp"
#include <cassert>
#include <fstream>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

// Helper to write a valid minimal uncompressed 24-bit BMP image
void createTestBmp(const fs::path& path, int width, int height) {
    std::ofstream out(path, std::ios::binary);
    int row_stride = (width * 3 + 3) & ~3;
    uint32_t image_size = static_cast<uint32_t>(row_stride * height);
    uint32_t file_size = 54 + image_size;

    uint8_t header[54] = {
        'B', 'M',
        static_cast<uint8_t>(file_size), static_cast<uint8_t>(file_size >> 8), static_cast<uint8_t>(file_size >> 16), static_cast<uint8_t>(file_size >> 24),
        0, 0, 0, 0,
        54, 0, 0, 0, // pixel offset
        40, 0, 0, 0, // header size
        static_cast<uint8_t>(width), static_cast<uint8_t>(width >> 8), static_cast<uint8_t>(width >> 16), static_cast<uint8_t>(width >> 24),
        static_cast<uint8_t>(height), static_cast<uint8_t>(height >> 8), static_cast<uint8_t>(height >> 16), static_cast<uint8_t>(height >> 24),
        1, 0, // planes
        24, 0, // bits per pixel
        0, 0, 0, 0, // compression (none)
        static_cast<uint8_t>(image_size), static_cast<uint8_t>(image_size >> 8), static_cast<uint8_t>(image_size >> 16), static_cast<uint8_t>(image_size >> 24),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    out.write(reinterpret_cast<const char*>(header), 54);

    std::vector<uint8_t> row(static_cast<size_t>(row_stride), 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            row[static_cast<size_t>(x * 3 + 0)] = 200; // Blue
            row[static_cast<size_t>(x * 3 + 1)] = 100; // Green
            row[static_cast<size_t>(x * 3 + 2)] = 50;  // Red
        }
        out.write(reinterpret_cast<const char*>(row.data()), row_stride);
    }
}

// Helper to write a valid minimal 16-bit PCM WAV file
void createTestWav(const fs::path& path, uint32_t sample_rate, double duration_sec, float frequency = 440.0f) {
    std::ofstream out(path, std::ios::binary);
    uint32_t num_samples = static_cast<uint32_t>(static_cast<double>(sample_rate) * duration_sec);
    uint32_t data_bytes = num_samples * 2;
    uint32_t total_file_size = 36 + data_bytes;

    out.write("RIFF", 4);
    uint32_t chunk_size = total_file_size - 8;
    out.write(reinterpret_cast<const char*>(&chunk_size), 4);
    out.write("WAVEfmt ", 8);

    uint32_t subchunk1_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t byte_rate = sample_rate * 2;
    uint16_t block_align = 2;
    uint16_t bits_per_sample = 16;

    out.write(reinterpret_cast<const char*>(&subchunk1_size), 4);
    out.write(reinterpret_cast<const char*>(&audio_format), 2);
    out.write(reinterpret_cast<const char*>(&num_channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out.write(reinterpret_cast<const char*>(&block_align), 2);
    out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_bytes), 4);

    for (uint32_t i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        int16_t sample = static_cast<int16_t>(std::sin(2.0f * 3.14159265f * frequency * t) * 16000.0f);
        out.write(reinterpret_cast<const char*>(&sample), 2);
    }
}

void testTextPreprocessing() {
    gemfinder::Logger::info("--- Test 1: Text Preprocessor ---");
    gemfinder::TextPreprocessor preprocessor;

    std::string text = "First paragraph.\nContains detailed weather forecast and precipitation information.\n\n"
                       "Second paragraph.\nExplains the architectural design of GemFinder in C++20.\n\n"
                       "Third paragraph with advanced indexing insights.";

    auto chunks = preprocessor.processString(text);
    assert(!chunks.empty());
    assert(chunks[0].start_line == 1);
    gemfinder::Logger::info("Text chunks: " + std::to_string(chunks.size()) +
                            ", Preview: '" + chunks[0].preview + "'");

    std::string doc_prompt = gemfinder::TextPreprocessor::formatDocumentPrompt("test.txt", chunks[0].text);
    assert(doc_prompt.starts_with("title: test.txt | text: "));
    std::string q_prompt = gemfinder::TextPreprocessor::formatQueryPrompt("weather forecast");
    assert(q_prompt == "task: search result | query: weather forecast");

    gemfinder::Logger::info("Test 1 completed successfully!");
}

void testImagePreprocessing() {
    gemfinder::Logger::info("--- Test 2: Image Preprocessor ---");
    fs::path bmp_path = "/tmp/test_image.bmp";
    createTestBmp(bmp_path, 32, 24);

    int w = 0, h = 0, c = 0;
    assert(gemfinder::ImagePreprocessor::getDimensions(bmp_path, w, h, c));
    assert(w == 32 && h == 24 && c == 3);

    gemfinder::ImagePreprocessor img_proc;
    auto processed = img_proc.processFile(bmp_path);
    assert(processed.has_value());
    assert(processed->original_width == 32);
    assert(processed->original_height == 24);
    assert(processed->target_width == 224);
    assert(processed->target_height == 224);
    // 224 * 224 * 3 = 150,528 float values
    assert(processed->rgb_planar_or_chw.size() == 224 * 224 * 3);
    assert(processed->rgb_planar_or_chw[0] >= 0.0f && processed->rgb_planar_or_chw[0] <= 1.0f);

    gemfinder::Logger::info("Image processed successfully: " + processed->preview_text);
    fs::remove(bmp_path);
    gemfinder::Logger::info("Test 2 completed successfully!");
}

void testAudioPreprocessing() {
    gemfinder::Logger::info("--- Test 3: Audio Preprocessor ---");
    fs::path wav_path = "/tmp/test_sine.wav";
    // 1.5 seconds at 44.1 kHz
    createTestWav(wav_path, 44100, 1.5, 440.0f);

    gemfinder::AudioPreprocessor aud_proc;
    auto processed = aud_proc.processFile(wav_path);
    assert(processed.has_value());
    assert(processed->sample_rate == 16000);
    assert(processed->channels == 1);
    assert(processed->total_duration_seconds > 1.4 && processed->total_duration_seconds < 1.6);
    assert(!processed->chunks.empty());
    // 1.5s at 16 kHz = 24,000 samples
    assert(processed->chunks[0].pcm_samples.size() >= 23000);

    gemfinder::Logger::info("Audio resampled to 16kHz mono: " + processed->chunks[0].preview_text +
                            " (Duration: " + std::to_string(processed->total_duration_seconds) + "s, Samples: " +
                            std::to_string(processed->chunks[0].pcm_samples.size()) + ")");

    fs::remove(wav_path);
    gemfinder::Logger::info("Test 3 completed successfully!");
}

void testPdfPreprocessing() {
    gemfinder::Logger::info("--- Test 4: PDF Digital Text Extraction ---");
    fs::path pdf_path = "/tmp/test_digital_gemfinder.pdf";

    // Generate digital vector PDF via groff
    int res = std::system("echo 'Test paragraph for digital vector PDF in GemFinder.' | groff -Tpdf > /tmp/test_digital_gemfinder.pdf 2>/dev/null");
    if (res != 0 || !fs::exists(pdf_path)) {
        gemfinder::Logger::warn("Unable to generate digital test PDF, skipping digital test");
        return;
    }

    gemfinder::PdfPreprocessor pdf_proc;
    auto chunks = pdf_proc.processFile(pdf_path);
    assert(!chunks.empty());
    assert(chunks[0].page_number == 1);
    assert(!chunks[0].is_ocr);
    assert(chunks[0].text.find("GemFinder") != std::string::npos);
    gemfinder::Logger::info("Extracted native PDF text: '" + chunks[0].preview + "'");

    fs::remove(pdf_path);
    gemfinder::Logger::info("Test 4 completed successfully!");
}

void testPdfOcrPreprocessing() {
    gemfinder::Logger::info("--- Test 5: PDF Scanned Image OCR Extraction ---");
    fs::path img_path = "/tmp/test_ocr_scan.png";
    fs::path pdf_path = "/tmp/test_ocr_scan.pdf";

    // Generate scanned PDF with text image
    int res = std::system("convert -size 800x200 xc:white -font DejaVu-Sans -pointsize 32 -fill black -draw \"text 50,100 'GemFinder OCR Test Document 2026'\" /tmp/test_ocr_scan.png 2>/dev/null && convert /tmp/test_ocr_scan.png /tmp/test_ocr_scan.pdf 2>/dev/null");
    if (res != 0 || !fs::exists(pdf_path)) {
        gemfinder::Logger::warn("Unable to generate raster PDF for OCR, skipping OCR test");
        return;
    }

    gemfinder::PdfPreprocessor pdf_proc;
    auto chunks = pdf_proc.processFile(pdf_path);
    assert(!chunks.empty());
    assert(chunks[0].page_number == 1);
    assert(chunks[0].is_ocr);
    assert(chunks[0].text.find("GemFinder") != std::string::npos || chunks[0].text.find("OCR") != std::string::npos);
    gemfinder::Logger::info("Extracted OCR text from scanned PDF: '" + chunks[0].preview + "'");

    fs::remove(img_path);
    fs::remove(pdf_path);
    gemfinder::Logger::info("Test 5 completed successfully!");
}

void testMediaRouter() {
    gemfinder::Logger::info("--- Test 6: Unified MediaPreprocessor Router ---");
    assert(gemfinder::MediaPreprocessor::detectModality("test.txt") == gemfinder::Modality::Text);
    assert(gemfinder::MediaPreprocessor::detectModality("document.pdf") == gemfinder::Modality::Text);
    assert(gemfinder::MediaPreprocessor::detectModality("photo.JPG") == gemfinder::Modality::Image);
    assert(gemfinder::MediaPreprocessor::detectModality("voice.wav") == gemfinder::Modality::Audio);
    assert(gemfinder::MediaPreprocessor::detectModality("movie.mp4") == gemfinder::Modality::Video);
    assert(gemfinder::MediaPreprocessor::detectModality("archive.zip") == gemfinder::Modality::Binary);

    fs::path temp_txt = "/tmp/router_test.md";
    {
        std::ofstream out(temp_txt);
        out << "# Document Header\nSample content to verify media routing.";
    }

    gemfinder::MediaPreprocessor router;
    auto payload = router.processFile(temp_txt);
    assert(payload.has_value());
    assert(payload->file.filename == "router_test.md");
    assert(payload->file.modality == gemfinder::Modality::Text);
    assert(payload->file.xxhash64 != 0);
    assert(!payload->chunks.empty());
    assert(payload->chunks[0].record.chunk_type == "text_paragraph");

    fs::remove(temp_txt);

    // Test routing PDF through MediaPreprocessor
    std::system("echo 'Test text for PDF MediaPreprocessor routing' | groff -Tpdf > /tmp/router_pdf.pdf 2>/dev/null");
    if (fs::exists("/tmp/router_pdf.pdf")) {
        auto pdf_payload = router.processFile("/tmp/router_pdf.pdf");
        assert(pdf_payload.has_value());
        assert(pdf_payload->file.filename == "router_pdf.pdf");
        assert(pdf_payload->file.modality == gemfinder::Modality::Text);
        assert(!pdf_payload->chunks.empty());
        assert(pdf_payload->chunks[0].record.chunk_type == "pdf_page");
        assert(pdf_payload->chunks[0].record.start_offset == 1);
        fs::remove("/tmp/router_pdf.pdf");
    }

    gemfinder::Logger::info("Test 6 completed successfully!");
}

int main() {
    gemfinder::Logger::info("=== Starting Multimodal Preprocessing Test Suite (Step 2) ===");
    testTextPreprocessing();
    testImagePreprocessing();
    testAudioPreprocessing();
    testPdfPreprocessing();
    testPdfOcrPreprocessing();
    testMediaRouter();
    gemfinder::Logger::info("=== ALL STEP 2 TESTS PASSED SUCCESSFULLY! ===");
    return 0;
}
