#pragma once

#include "common/Types.hpp"
#include <string>
#include <vector>
#include <filesystem>

namespace gemfinder {

namespace fs = std::filesystem;

inline constexpr std::string_view kDefaultSocketPath = "/tmp/gemfinder.sock";

enum class IpcAction {
    Ping,
    SearchText,
    SearchImage,
    SearchAudio,
    IndexDirectory,
    GetStats,
    Unknown
};

[[nodiscard]] inline std::string actionToString(IpcAction action) {
    switch (action) {
        case IpcAction::Ping:           return "ping";
        case IpcAction::SearchText:     return "search_text";
        case IpcAction::SearchImage:    return "search_image";
        case IpcAction::SearchAudio:    return "search_audio";
        case IpcAction::IndexDirectory: return "index_dir";
        case IpcAction::GetStats:       return "get_stats";
        case IpcAction::Unknown:        return "unknown";
    }
    return "unknown";
}

[[nodiscard]] inline IpcAction stringToAction(std::string_view sv) {
    if (sv == "ping")         return IpcAction::Ping;
    if (sv == "search_text")  return IpcAction::SearchText;
    if (sv == "search_image") return IpcAction::SearchImage;
    if (sv == "search_audio") return IpcAction::SearchAudio;
    if (sv == "index_dir")    return IpcAction::IndexDirectory;
    if (sv == "get_stats")    return IpcAction::GetStats;
    return IpcAction::Unknown;
}

struct IpcRequest {
    IpcAction action{IpcAction::Unknown};
    std::string query;
    std::string path;
    size_t limit{10};
};

struct IpcResponse {
    bool success{false};
    std::string error_message;
    double latency_ms{0.0};
    std::vector<SearchResult> results;
    size_t total_files{0};
    size_t total_chunks{0};
    std::string model_name;
};

// Serialization / Deserialization helpers (delimiter-based or line-based for ultra-low latency)
std::string serializeRequest(const IpcRequest& req);
bool deserializeRequest(std::string_view raw, IpcRequest& out_req);

std::string serializeResponse(const IpcResponse& resp);
bool deserializeResponse(std::string_view raw, IpcResponse& out_resp);

} // namespace gemfinder
