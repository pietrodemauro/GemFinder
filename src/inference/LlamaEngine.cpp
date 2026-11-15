#include "inference/LlamaEngine.hpp"
#include "inference/Normalization.hpp"
#include "common/Logger.hpp"

#include <llama.h>
#include <ggml-backend.h>
#include <mtmd.h>
#include <mtmd-image.h>
#include <thread>
#include <algorithm>
#include <cstring>
#include <cmath>

namespace gemfinder {

namespace {

int resolveDefaultThreads(int n_threads) {
    if (n_threads > 0) {
        return n_threads;
    }
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw <= 1) return 1;
    // Limit to half of maximum cores by default to keep the desktop responsive
    return static_cast<int>(hw / 2);
}

} // namespace

LlamaEngine::LlamaEngine(size_t target_dimension, int n_threads, bool use_gpu)
    : target_dim_(target_dimension),
      n_threads_(resolveDefaultThreads(n_threads)),
      use_gpu_(use_gpu) {
    // Suppress verbose llama.cpp and mtmd debug/info spam on stderr
    auto log_cb = [](ggml_log_level level, const char* text, void* user_data) {
        (void)user_data;
        if (level >= GGML_LOG_LEVEL_ERROR) {
            std::cerr << text;
        }
    };
    llama_log_set(log_cb, nullptr);
    mtmd_log_set(log_cb, nullptr);
    llama_backend_init();
    ggml_backend_load_all();

    size_t n_devs = ggml_backend_dev_count();
    std::string dev_info;
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev) {
            if (!dev_info.empty()) dev_info += ", ";
            dev_info += std::string(ggml_backend_dev_name(dev)) + " (" + ggml_backend_dev_description(dev) + ")";
        }
    }
    Logger::info("LlamaEngine: Compute devices detected: " + (dev_info.empty() ? "CPU fallback" : dev_info));

    Logger::info("LlamaEngine: Allocated " + std::to_string(n_threads_) +
                 " CPU threads (total hardware cores: " +
                 std::to_string(std::thread::hardware_concurrency()) + ")");
}

LlamaEngine::~LlamaEngine() {
    unload();
}

LlamaEngine::LlamaEngine(LlamaEngine&& other) noexcept {
    std::lock_guard<std::mutex> lock(other.mutex_);
    model_ = other.model_;
    ctx_ = other.ctx_;
    ctx_mtmd_ = other.ctx_mtmd_;
    target_dim_ = other.target_dim_;
    n_threads_ = other.n_threads_;
    use_gpu_ = other.use_gpu_;
    model_name_ = std::move(other.model_name_);

    other.model_ = nullptr;
    other.ctx_ = nullptr;
    other.ctx_mtmd_ = nullptr;
}

LlamaEngine& LlamaEngine::operator=(LlamaEngine&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(mutex_, other.mutex_);
        unload();
        model_ = other.model_;
        ctx_ = other.ctx_;
        ctx_mtmd_ = other.ctx_mtmd_;
        target_dim_ = other.target_dim_;
        n_threads_ = other.n_threads_;
        use_gpu_ = other.use_gpu_;
        model_name_ = std::move(other.model_name_);

        other.model_ = nullptr;
        other.ctx_ = nullptr;
        other.ctx_mtmd_ = nullptr;
    }
    return *this;
}

void LlamaEngine::unload() {
    if (ctx_mtmd_) {
        mtmd_free(ctx_mtmd_);
        ctx_mtmd_ = nullptr;
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

bool LlamaEngine::loadModel(const fs::path& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    unload();

    if (!fs::exists(model_path)) {
        Logger::error("Model file does not exist: " + model_path.string());
        return false;
    }

    llama_model_params mparams = llama_model_default_params();
    // Offload all layers to GPU (Vulkan/CUDA) if GPU is requested
    mparams.n_gpu_layers = use_gpu_ ? 99 : 0;

    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model_) {
        Logger::error("llama_model_load_from_file failed for " + model_path.string());
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    cparams.n_threads = n_threads_;
    cparams.n_threads_batch = n_threads_;

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        Logger::error("llama_init_from_model failed for " + model_path.string());
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    model_name_ = model_path.filename().string();
    Logger::info("Successfully loaded GGUF model: " + model_name_ +
                 " (Embedding Raw Dim: " + std::to_string(rawModelDimension()) +
                 ", Target MRL Dim: " + std::to_string(target_dim_) + ")");

    // Attempt to load multimodal vision/audio projector (mmproj)
    loadVisionProjector(model_path);

    return true;
}

bool LlamaEngine::loadVisionProjector(const fs::path& model_path) {
    fs::path mmproj_path;
    std::vector<fs::path> candidates = {
        model_path.parent_path() / "mmproj.gguf",
        model_path.parent_path() / "mmproj-BF16.gguf",
        model_path.parent_path() / "mmproj-Q8_0.gguf",
        fs::current_path() / "models" / "mmproj.gguf",
        fs::current_path() / "models" / "mmproj-BF16.gguf"
    };
    const char* home = std::getenv("HOME");
    if (home) {
        candidates.push_back(fs::path(home) / ".local/share/gemfinder/models/mmproj.gguf");
        candidates.push_back(fs::path(home) / ".local/share/gemfinder/models/mmproj-BF16.gguf");
    }
    for (const auto& c : candidates) {
        if (!c.empty() && fs::exists(c)) {
            mmproj_path = c;
            break;
        }
    }

    if (mmproj_path.empty()) {
        Logger::info("LlamaEngine: No mmproj (vision projector) file found.");
        return false;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = use_gpu_;
    mparams.n_threads = n_threads_;
    ctx_mtmd_ = mtmd_init_from_file(mmproj_path.string().c_str(), model_, mparams);
    if (!ctx_mtmd_) {
        Logger::warn("LlamaEngine: mtmd_init_from_file failed for " + mmproj_path.string());
        return false;
    }

    Logger::info("LlamaEngine: Successfully loaded vision projector: " + mmproj_path.filename().string() +
                 " (Vision: " + (mtmd_support_vision(ctx_mtmd_) ? "YES" : "NO") +
                 ", Audio: " + (mtmd_support_audio(ctx_mtmd_) ? "YES" : "NO") + ")");
    return true;
}

bool LlamaEngine::hasVisionSupport() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return (ctx_mtmd_ != nullptr && mtmd_support_vision(ctx_mtmd_));
}

bool LlamaEngine::isLoaded() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return (model_ != nullptr && ctx_ != nullptr);
}

std::string LlamaEngine::modelName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return model_name_.empty() ? "None" : model_name_;
}

size_t LlamaEngine::rawModelDimension() const noexcept {
    if (!model_) return 0;
    return static_cast<size_t>(llama_model_n_embd(model_));
}

Embedding LlamaEngine::embedText(std::string_view text) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_ || !ctx_ || text.empty()) {
        return Embedding(target_dim_, 0.0f);
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens(text.size() + 16);

    int n_tokens = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,  // add_special
        true   // parse_special
    );

    if (n_tokens < 0) {
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = llama_tokenize(
            vocab,
            text.data(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            true
        );
    }

    if (n_tokens <= 0) {
        return Embedding(target_dim_, 0.0f);
    }

    // Encoder requires n_ubatch >= n_tokens. Clamp n_tokens to context & batch capacity.
    int max_tokens = 2048;
    if (ctx_) {
        int ctx_lim = static_cast<int>(llama_n_ctx(ctx_));
        int batch_lim = static_cast<int>(llama_n_batch(ctx_));
        if (ctx_lim > 0 && batch_lim > 0) {
            max_tokens = std::min(ctx_lim, batch_lim);
        }
    }
    if (n_tokens > max_tokens) {
        n_tokens = max_tokens;
    }
    tokens.resize(static_cast<size_t>(n_tokens));

    // Prepare batch with output enabled on all tokens for mean pooling
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[static_cast<size_t>(i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1; // Mark output for all tokens for mean pooling
    }
    batch.n_tokens = n_tokens;

    int res = 0;
    if (llama_model_has_encoder(model_)) {
        res = llama_encode(ctx_, batch);
    } else {
        res = llama_decode(ctx_, batch);
    }
    llama_batch_free(batch);

    if (res != 0) {
        Logger::error("llama execution failed during embedding generation");
        return Embedding(target_dim_, 0.0f);
    }

    // Retrieve embeddings
    float* embd = llama_get_embeddings_seq(ctx_, 0);
    if (!embd) {
        embd = llama_get_embeddings_ith(ctx_, -1);
    }
    if (!embd) {
        embd = llama_get_embeddings(ctx_);
    }

    if (!embd) {
        Logger::error("Failed to retrieve embeddings from llama context");
        return Embedding(target_dim_, 0.0f);
    }

    size_t raw_dim = rawModelDimension();
    return applyMrlAndNormalize(std::span<const float>(embd, raw_dim), target_dim_);
}

Embedding LlamaEngine::embedImage(std::span<const float> rgb_pixels, int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_ || !ctx_ || rgb_pixels.empty() || width <= 0 || height <= 0) {
        return Embedding(target_dim_, 0.0f);
    }

    if (!ctx_mtmd_ || !mtmd_support_vision(ctx_mtmd_)) {
        Logger::warn("LlamaEngine: mtmd visual encoder unavailable, fallback to text embedding.");
        return embedText("<|image|>");
    }

    // Convert normalized [0.0, 1.0] float RGB pixels to 8-bit unsigned char RGBRGB...
    std::vector<unsigned char> u8_pixels(static_cast<size_t>(width * height * 3));
    for (size_t i = 0; i < u8_pixels.size() && i < rgb_pixels.size(); ++i) {
        float val = rgb_pixels[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        u8_pixels[i] = static_cast<unsigned char>(std::clamp(static_cast<int>(std::round(val * 255.0f)), 0, 255));
    }

    mtmd_bitmap* bmp = mtmd_bitmap_init(
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        u8_pixels.data()
    );
    if (!bmp) {
        Logger::error("LlamaEngine: Unable to create mtmd_bitmap for image.");
        return Embedding(target_dim_, 0.0f);
    }

    const std::string marker = mtmd_get_marker(ctx_mtmd_);
    mtmd_input_text text{ marker.c_str(), marker.size(), false, true };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bptr = bmp;
    int32_t tok_res = mtmd_tokenize(ctx_mtmd_, chunks, &text, &bptr, 1);
    mtmd_bitmap_free(bmp);

    if (tok_res != 0) {
        Logger::error("LlamaEngine: mtmd_tokenize failed (error " + std::to_string(tok_res) + ")");
        mtmd_input_chunks_free(chunks);
        return Embedding(target_dim_, 0.0f);
    }

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    const mtmd_input_chunk* img_chunk = nullptr;
    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            img_chunk = ch;
            break;
        }
    }

    if (!img_chunk) {
        Logger::error("LlamaEngine: No image chunk found after tokenization.");
        mtmd_input_chunks_free(chunks);
        return Embedding(target_dim_, 0.0f);
    }

    int32_t enc_res = mtmd_encode_chunk(ctx_mtmd_, img_chunk);
    if (enc_res != 0) {
        Logger::error("LlamaEngine: mtmd_encode_chunk failed for image (error " + std::to_string(enc_res) + ")");
        mtmd_input_chunks_free(chunks);
        return Embedding(target_dim_, 0.0f);
    }

    float* embd_tokens = mtmd_get_output_embd(ctx_mtmd_);
    size_t n_img_tokens = mtmd_input_chunk_get_n_tokens(img_chunk);
    size_t embd_dim = static_cast<size_t>(llama_model_n_embd_inp(model_));

    if (!embd_tokens || n_img_tokens == 0 || embd_dim == 0) {
        Logger::error("LlamaEngine: Invalid output embeddings from visual encoder.");
        mtmd_input_chunks_free(chunks);
        return Embedding(target_dim_, 0.0f);
    }

    // Try decoding through transformer context for contextual multimodal representation
    bool eval_ok = false;
    float* embd_out = nullptr;

    llama_batch batch = llama_batch_init(static_cast<int32_t>(n_img_tokens), static_cast<int32_t>(embd_dim), 1);
    batch.n_tokens = static_cast<int32_t>(n_img_tokens);
    for (size_t i = 0; i < n_img_tokens; ++i) {
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1; // enable pooling on all visual tokens
    }
    std::memcpy(batch.embd, embd_tokens, n_img_tokens * embd_dim * sizeof(float));

    int res = 0;
    if (llama_model_has_encoder(model_)) {
        res = llama_encode(ctx_, batch);
    } else {
        res = llama_decode(ctx_, batch);
    }
    llama_batch_free(batch);
    mtmd_input_chunks_free(chunks);

    if (res == 0) {
        embd_out = llama_get_embeddings_seq(ctx_, 0);
        if (!embd_out) {
            embd_out = llama_get_embeddings_ith(ctx_, -1);
        }
        if (!embd_out) {
            embd_out = llama_get_embeddings(ctx_);
        }
        if (embd_out) {
            eval_ok = true;
        }
    }

    if (eval_ok && embd_out) {
        size_t raw_dim = rawModelDimension();
        return applyMrlAndNormalize(std::span<const float>(embd_out, raw_dim), target_dim_);
    }

    // Fallback: direct spatial mean-pooling over visual encoder output tokens
    std::vector<float> pooled(embd_dim, 0.0f);
    for (size_t i = 0; i < n_img_tokens; ++i) {
        for (size_t d = 0; d < embd_dim; ++d) {
            pooled[d] += embd_tokens[i * embd_dim + d];
        }
    }
    for (size_t d = 0; d < embd_dim; ++d) {
        pooled[d] /= static_cast<float>(n_img_tokens);
    }
    return applyMrlAndNormalize(pooled, target_dim_);
}

Embedding LlamaEngine::embedAudio(std::span<const float> pcm_samples, uint32_t sample_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_ || !ctx_ || pcm_samples.empty()) {
        return Embedding(target_dim_, 0.0f);
    }

    if (!ctx_mtmd_ || !mtmd_support_audio(ctx_mtmd_)) {
        return embedText("<|audio|>");
    }

    mtmd_bitmap* bmp = mtmd_bitmap_init_from_audio(pcm_samples.size(), pcm_samples.data());
    if (!bmp) {
        return embedText("<|audio|>");
    }

    const std::string marker = mtmd_get_marker(ctx_mtmd_);
    mtmd_input_text text{ marker.c_str(), marker.size(), false, true };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bptr = bmp;
    int32_t tok_res = mtmd_tokenize(ctx_mtmd_, chunks, &text, &bptr, 1);
    mtmd_bitmap_free(bmp);

    if (tok_res != 0) {
        mtmd_input_chunks_free(chunks);
        return embedText("<|audio|>");
    }

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    const mtmd_input_chunk* aud_chunk = nullptr;
    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            aud_chunk = ch;
            break;
        }
    }

    if (!aud_chunk) {
        mtmd_input_chunks_free(chunks);
        return embedText("<|audio|>");
    }

    int32_t enc_res = mtmd_encode_chunk(ctx_mtmd_, aud_chunk);
    if (enc_res != 0) {
        mtmd_input_chunks_free(chunks);
        return embedText("<|audio|>");
    }

    float* embd_tokens = mtmd_get_output_embd(ctx_mtmd_);
    size_t n_aud_tokens = mtmd_input_chunk_get_n_tokens(aud_chunk);
    size_t embd_dim = static_cast<size_t>(llama_model_n_embd_inp(model_));
    mtmd_input_chunks_free(chunks);

    if (!embd_tokens || n_aud_tokens == 0 || embd_dim == 0) {
        return embedText("<|audio|>");
    }

    std::vector<float> pooled(embd_dim, 0.0f);
    for (size_t i = 0; i < n_aud_tokens; ++i) {
        for (size_t d = 0; d < embd_dim; ++d) {
            pooled[d] += embd_tokens[i * embd_dim + d];
        }
    }
    for (size_t d = 0; d < embd_dim; ++d) {
        pooled[d] /= static_cast<float>(n_aud_tokens);
    }
    return applyMrlAndNormalize(pooled, target_dim_);
}

} // namespace gemfinder
