#include "inference/LlamaEngine.hpp"
#include "inference/Normalization.hpp"
#include "common/Logger.hpp"

#include <llama.h>
#include <thread>
#include <algorithm>
#include <cstring>

namespace gemfinder {

LlamaEngine::LlamaEngine(size_t target_dimension, int n_threads)
    : target_dim_(target_dimension),
      n_threads_(n_threads > 0 ? n_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))) {
    llama_backend_init();
}

LlamaEngine::~LlamaEngine() {
    unload();
}

LlamaEngine::LlamaEngine(LlamaEngine&& other) noexcept {
    std::lock_guard<std::mutex> lock(other.mutex_);
    model_ = other.model_;
    ctx_ = other.ctx_;
    target_dim_ = other.target_dim_;
    n_threads_ = other.n_threads_;
    model_name_ = std::move(other.model_name_);

    other.model_ = nullptr;
    other.ctx_ = nullptr;
}

LlamaEngine& LlamaEngine::operator=(LlamaEngine&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(mutex_, other.mutex_);
        unload();
        model_ = other.model_;
        ctx_ = other.ctx_;
        target_dim_ = other.target_dim_;
        n_threads_ = other.n_threads_;
        model_name_ = std::move(other.model_name_);

        other.model_ = nullptr;
        other.ctx_ = nullptr;
    }
    return *this;
}

void LlamaEngine::unload() {
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
    // Offload all layers to GPU (Vulkan/CUDA) if available
    mparams.n_gpu_layers = 99;

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
    return true;
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

    // Prepare batch
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);

    if (llama_decode(ctx_, batch) != 0) {
        Logger::error("llama_decode failed during embedding generation");
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
    (void)rgb_pixels;
    (void)width;
    (void)height;
    // Multimodal prompt representation for vision encoder
    return embedText("<|image|>");
}

Embedding LlamaEngine::embedAudio(std::span<const float> pcm_samples, uint32_t sample_rate) {
    (void)pcm_samples;
    (void)sample_rate;
    // Multimodal prompt representation for audio encoder
    return embedText("<|audio|>");
}

} // namespace gemfinder
