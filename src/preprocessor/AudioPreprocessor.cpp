#include "preprocessor/AudioPreprocessor.hpp"
#include "common/Logger.hpp"

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#include <miniaudio/miniaudio.h>

#include <cmath>
#include <iomanip>
#include <sstream>

namespace gemfinder {

namespace {

std::string formatTime(int64_t seconds) {
    int64_t m = seconds / 60;
    int64_t s = seconds % 60;
    std::ostringstream ss;
    ss << std::setw(2) << std::setfill('0') << m << ":"
       << std::setw(2) << std::setfill('0') << s;
    return ss.str();
}

} // namespace

AudioPreprocessor::AudioPreprocessor() : config_{} {}
AudioPreprocessor::AudioPreprocessor(Config config) : config_(config) {}

std::optional<ProcessedAudio> AudioPreprocessor::processFile(const fs::path& path) const {
    ma_decoder_config decoderConfig = ma_decoder_config_init(ma_format_f32, 1, config_.target_sample_rate);
    ma_decoder decoder;

    ma_result result = ma_decoder_init_file(path.c_str(), &decoderConfig, &decoder);
    if (result != MA_SUCCESS) {
        Logger::error("miniaudio failed to open file: " + path.string() + " (code: " + std::to_string(result) + ")");
        return std::nullopt;
    }

    std::vector<float> all_samples;
    constexpr size_t kChunkReadFrames = 4096;
    std::vector<float> temp_buffer(kChunkReadFrames);

    ma_uint64 framesRead = 0;
    while (true) {
        ma_result readResult = ma_decoder_read_pcm_frames(&decoder, temp_buffer.data(), kChunkReadFrames, &framesRead);
        if (framesRead > 0) {
            all_samples.insert(all_samples.end(), temp_buffer.begin(), temp_buffer.begin() + static_cast<std::ptrdiff_t>(framesRead));
        }
        if (readResult != MA_SUCCESS || framesRead == 0) {
            break;
        }
    }

    ma_decoder_uninit(&decoder);

    if (all_samples.empty()) {
        Logger::warn("AudioPreprocessor: 0 frames read from " + path.string());
        return std::nullopt;
    }

    ProcessedAudio audio;
    audio.sample_rate = config_.target_sample_rate;
    audio.channels = 1;
    audio.total_duration_seconds = static_cast<double>(all_samples.size()) / static_cast<double>(config_.target_sample_rate);

    const size_t window_samples = static_cast<size_t>(config_.window_duration_seconds * config_.target_sample_rate);
    const size_t overlap_samples = static_cast<size_t>(config_.overlap_duration_seconds * config_.target_sample_rate);
    const size_t step_samples = (window_samples > overlap_samples) ? (window_samples - overlap_samples) : window_samples;

    if (all_samples.size() <= window_samples) {
        AudioChunk chunk;
        chunk.start_second = 0;
        chunk.end_second = static_cast<int64_t>(std::ceil(audio.total_duration_seconds));
        chunk.sample_offset = 0;
        chunk.sample_count = all_samples.size();
        chunk.pcm_samples = std::move(all_samples);
        chunk.preview_text = "Audio [" + formatTime(0) + " - " + formatTime(chunk.end_second) + "] " + path.filename().string();
        chunk.document_prompt = "title: " + path.filename().string() + " | <|audio|>";
        audio.chunks.push_back(std::move(chunk));
    } else {
        size_t cursor = 0;
        while (cursor < all_samples.size()) {
            size_t end = std::min(cursor + window_samples, all_samples.size());
            size_t count = end - cursor;

            AudioChunk chunk;
            chunk.sample_offset = cursor;
            chunk.sample_count = count;
            chunk.start_second = static_cast<int64_t>(cursor / config_.target_sample_rate);
            chunk.end_second = static_cast<int64_t>(std::ceil(static_cast<double>(end) / config_.target_sample_rate));
            chunk.pcm_samples.assign(all_samples.begin() + static_cast<std::ptrdiff_t>(cursor),
                                     all_samples.begin() + static_cast<std::ptrdiff_t>(end));
            chunk.preview_text = "Audio [" + formatTime(chunk.start_second) + " - " + formatTime(chunk.end_second) + "] " + path.filename().string();
            chunk.document_prompt = "title: " + path.filename().string() + " (at " + std::to_string(chunk.start_second) + "s) | <|audio|>";

            audio.chunks.push_back(std::move(chunk));

            if (end >= all_samples.size()) {
                break;
            }
            cursor += step_samples;
        }
    }

    return audio;
}

std::optional<ProcessedAudio> AudioPreprocessor::processMemory(const void* data, size_t length, const std::string& filename) const {
    if (!data || length == 0) return std::nullopt;

    ma_decoder_config decoderConfig = ma_decoder_config_init(ma_format_f32, 1, config_.target_sample_rate);
    ma_decoder decoder;

    ma_result result = ma_decoder_init_memory(data, length, &decoderConfig, &decoder);
    if (result != MA_SUCCESS) {
        return std::nullopt;
    }

    std::vector<float> all_samples;
    constexpr size_t kChunkReadFrames = 4096;
    std::vector<float> temp_buffer(kChunkReadFrames);

    ma_uint64 framesRead = 0;
    while (true) {
        ma_result readResult = ma_decoder_read_pcm_frames(&decoder, temp_buffer.data(), kChunkReadFrames, &framesRead);
        if (framesRead > 0) {
            all_samples.insert(all_samples.end(), temp_buffer.begin(), temp_buffer.begin() + static_cast<std::ptrdiff_t>(framesRead));
        }
        if (readResult != MA_SUCCESS || framesRead == 0) {
            break;
        }
    }

    ma_decoder_uninit(&decoder);

    if (all_samples.empty()) {
        return std::nullopt;
    }

    ProcessedAudio audio;
    audio.sample_rate = config_.target_sample_rate;
    audio.channels = 1;
    audio.total_duration_seconds = static_cast<double>(all_samples.size()) / static_cast<double>(config_.target_sample_rate);

    AudioChunk chunk;
    chunk.start_second = 0;
    chunk.end_second = static_cast<int64_t>(std::ceil(audio.total_duration_seconds));
    chunk.sample_offset = 0;
    chunk.sample_count = all_samples.size();
    chunk.pcm_samples = std::move(all_samples);
    chunk.preview_text = "Audio [" + formatTime(0) + " - " + formatTime(chunk.end_second) + "] " + filename;
    chunk.document_prompt = "title: " + filename + " | <|audio|>";
    audio.chunks.push_back(std::move(chunk));

    return audio;
}

} // namespace gemfinder
