#include "common/IpcProtocol.hpp"
#include <sstream>

namespace gemfinder {

namespace {

std::vector<std::string> splitTabs(std::string_view line) {
    std::vector<std::string> tokens;
    size_t start = 0;
    while (start < line.size()) {
        size_t pos = line.find('\t', start);
        if (pos == std::string_view::npos) {
            tokens.emplace_back(line.substr(start));
            break;
        }
        tokens.emplace_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return tokens;
}

std::string sanitizeString(std::string_view sv) {
    std::string s;
    s.reserve(sv.size());
    for (char c : sv) {
        if (c == '\t') s += ' ';
        else if (c == '\n' || c == '\r') s += ' ';
        else s += c;
    }
    return s;
}

} // namespace

std::string serializeRequest(const IpcRequest& req) {
    std::string out = actionToString(req.action);
    switch (req.action) {
        case IpcAction::SearchText:
            out += "\t" + std::to_string(req.limit) + "\t" + sanitizeString(req.query);
            break;
        case IpcAction::SearchImage:
            out += "\t" + std::to_string(req.limit) + "\t" + sanitizeString(req.path);
            break;
        case IpcAction::SearchAudio:
            out += "\t" + std::to_string(req.limit) + "\t" + sanitizeString(req.path);
            break;
        case IpcAction::IndexDirectory:
            out += "\t" + sanitizeString(req.path);
            break;
        default:
            break;
    }
    out += "\n";
    return out;
}

bool deserializeRequest(std::string_view raw, IpcRequest& out_req) {
    if (raw.empty()) return false;
    if (raw.back() == '\n') raw.remove_suffix(1);
    if (!raw.empty() && raw.back() == '\r') raw.remove_suffix(1);

    auto parts = splitTabs(raw);
    if (parts.empty()) return false;

    out_req.action = stringToAction(parts[0]);
    if (out_req.action == IpcAction::Unknown) return false;

    if (out_req.action == IpcAction::SearchText && parts.size() >= 3) {
        try { out_req.limit = std::stoul(parts[1]); } catch (...) { out_req.limit = 10; }
        out_req.query = parts[2];
    } else if ((out_req.action == IpcAction::SearchImage || out_req.action == IpcAction::SearchAudio) && parts.size() >= 3) {
        try { out_req.limit = std::stoul(parts[1]); } catch (...) { out_req.limit = 10; }
        out_req.path = parts[2];
    } else if (out_req.action == IpcAction::IndexDirectory && parts.size() >= 2) {
        out_req.path = parts[1];
    }

    return true;
}

std::string serializeResponse(const IpcResponse& resp) {
    std::string out;
    if (!resp.success) {
        out = "ERR\t" + sanitizeString(resp.error_message) + "\n";
        return out;
    }

    out = "OK\t" + std::to_string(resp.latency_ms);

    if (resp.total_files > 0 || !resp.model_name.empty()) {
        // Stats response
        out += "\tSTATS\t" + std::to_string(resp.total_files) +
               "\t" + std::to_string(resp.total_chunks) +
               "\t" + sanitizeString(resp.model_name) + "\n";
        return out;
    }

    // Results response
    out += "\tRESULTS\t" + std::to_string(resp.results.size()) + "\n";
    for (const auto& r : resp.results) {
        out += std::to_string(r.file_id) + "\t" +
               std::to_string(r.chunk_id) + "\t" +
               sanitizeString(r.filepath) + "\t" +
               sanitizeString(r.filename) + "\t" +
               std::to_string(r.score) + "\t" +
               std::to_string(r.start_offset) + "\t" +
               sanitizeString(r.chunk_type) + "\t" +
               std::to_string(r.rank_vector) + "\t" +
               std::to_string(r.rank_lexical) + "\t" +
               sanitizeString(r.preview_text) + "\n";
    }

    return out;
}

bool deserializeResponse(std::string_view raw, IpcResponse& out_resp) {
    out_resp = IpcResponse{};
    if (raw.empty()) return false;

    std::istringstream stream{std::string(raw)};
    std::string first_line;
    if (!std::getline(stream, first_line)) return false;

    auto header = splitTabs(first_line);
    if (header.empty()) return false;

    if (header[0] == "ERR") {
        out_resp.success = false;
        if (header.size() > 1) out_resp.error_message = header[1];
        return true;
    }

    if (header[0] != "OK") return false;

    out_resp.success = true;
    if (header.size() > 1) {
        try { out_resp.latency_ms = std::stod(header[1]); } catch (...) {}
    }

    if (header.size() > 2 && header[2] == "STATS" && header.size() >= 6) {
        try { out_resp.total_files = std::stoul(header[3]); } catch (...) {}
        try { out_resp.total_chunks = std::stoul(header[4]); } catch (...) {}
        out_resp.model_name = header[5];
        return true;
    }

    if (header.size() > 2 && header[2] == "RESULTS") {
        size_t count = 0;
        if (header.size() > 3) {
            try { count = std::stoul(header[3]); } catch (...) {}
        }

        std::string row_line;
        while (std::getline(stream, row_line) && out_resp.results.size() < count) {
            if (row_line.empty()) continue;
            auto cols = splitTabs(row_line);
            if (cols.size() >= 10) {
                SearchResult sr;
                try {
                    sr.file_id = std::stoll(cols[0]);
                    sr.chunk_id = std::stoll(cols[1]);
                    sr.filepath = cols[2];
                    sr.filename = cols[3];
                    sr.score = std::stod(cols[4]);
                    sr.start_offset = std::stoll(cols[5]);
                    sr.chunk_type = cols[6];
                    sr.rank_vector = std::stoi(cols[7]);
                    sr.rank_lexical = std::stoi(cols[8]);
                    sr.preview_text = cols[9];
                } catch (...) {
                    continue;
                }
                out_resp.results.push_back(std::move(sr));
            }
        }
        return true;
    }

    return true;
}

} // namespace gemfinder
