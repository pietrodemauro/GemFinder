#include "common/Version.hpp"
#include "common/Logger.hpp"
#include "common/Hasher.hpp"
#include "common/Types.hpp"
#include "preprocessor/MediaPreprocessor.hpp"
#include "inference/EngineFactory.hpp"
#include "inference/InferenceEngine.hpp"
#include <sqlite3.h>
#include <iostream>
#include <string_view>
#include <filesystem>
#include <chrono>

void printUsage(std::string_view progName) {
    std::cout << progName << " - " << gemfinder::kProjectDescription << "\n\n"
              << "Uso:\n"
              << "  " << progName << " --version           Mostra la versione del software\n"
              << "  " << progName << " --help              Mostra questo messaggio di aiuto\n"
              << "  " << progName << " --info              Mostra le librerie e lo stato del sistema\n"
              << "  " << progName << " --hash <file>       Calcola l'hash xxHash64 di un file\n"
              << "  " << progName << " --inspect <file>    Pre-processa un file e mostra i chunk estratti\n"
              << "  " << progName << " --embed <testo> [--model <path.gguf>]  Calcola l'embedding di un testo\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string_view arg1 = argv[1];

    if (arg1 == "--version" || arg1 == "-v") {
        std::cout << gemfinder::kProjectName << " versione " << gemfinder::kVersion << "\n";
        return 0;
    }

    if (arg1 == "--help" || arg1 == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    if (arg1 == "--info") {
        std::cout << "=== GemFinder System Info ===\n";
        std::cout << "Progetto:   " << gemfinder::kProjectName << " (" << gemfinder::kVersion << ")\n";
        std::cout << "C++ Std:    " << __cplusplus << "\n";
        std::cout << "SQLite:     " << sqlite3_libversion() << " (source: " << sqlite3_sourceid() << ")\n";
        std::cout << "Vector Dim: " << gemfinder::kEmbeddingDim << " (Matryoshka L2 Normalizzato)\n";
        std::cout << "Raw Dim:    " << gemfinder::kRawEmbeddingDim << "\n";
        return 0;
    }

    if (arg1 == "--hash" && argc >= 3) {
        std::filesystem::path target = argv[2];
        if (!std::filesystem::exists(target)) {
            gemfinder::Logger::error("File non trovato: " + target.string());
            return 1;
        }
        uint64_t hash = gemfinder::Hasher::hashFile(target);
        std::cout << "File:   " << target << "\n";
        std::cout << "XXH64:  0x" << std::hex << hash << std::dec << " (" << hash << ")\n";
        return 0;
    }

    if (arg1 == "--inspect" && argc >= 3) {
        std::filesystem::path target = argv[2];
        if (!std::filesystem::exists(target)) {
            gemfinder::Logger::error("File non trovato: " + target.string());
            return 1;
        }

        gemfinder::MediaPreprocessor router;
        auto payload = router.processFile(target);
        if (!payload) {
            gemfinder::Logger::error("Impossibile pre-processare il file: " + target.string());
            return 1;
        }

        std::cout << "=== GemFinder File Inspection ===\n";
        std::cout << "Path:      " << payload->file.filepath << "\n";
        std::cout << "Filename:  " << payload->file.filename << "\n";
        std::cout << "Modality:  " << gemfinder::modalityToString(payload->file.modality) << "\n";
        std::cout << "Size:      " << payload->file.file_size << " bytes\n";
        std::cout << "XXH64:     0x" << std::hex << payload->file.xxhash64 << std::dec << "\n";
        std::cout << "Chunks:    " << payload->chunks.size() << "\n\n";

        for (size_t i = 0; i < payload->chunks.size(); ++i) {
            const auto& chunk = payload->chunks[i];
            std::cout << "  [Chunk " << (i + 1) << "]\n";
            std::cout << "    Tipo:    " << chunk.record.chunk_type << "\n";
            std::cout << "    Offset:  " << chunk.record.start_offset << " -> " << chunk.record.end_offset << "\n";
            std::cout << "    Preview: " << chunk.record.preview_text << "\n";
            if (!chunk.raw_media_buffer.empty()) {
                std::cout << "    Tensor:  " << chunk.raw_media_buffer.size() << " floats\n";
            }
            std::cout << "    Prompt:  " << chunk.prompt.substr(0, std::min<size_t>(100, chunk.prompt.size()))
                      << (chunk.prompt.size() > 100 ? "..." : "") << "\n\n";
        }
        return 0;
    }

    if (arg1 == "--embed" && argc >= 3) {
        std::string_view text = argv[2];
        std::optional<std::filesystem::path> model_path = std::nullopt;
        for (int i = 3; i < argc; ++i) {
            if (std::string_view(argv[i]) == "--model" && i + 1 < argc) {
                model_path = argv[i + 1];
                i++;
            }
        }

        auto start_t = std::chrono::high_resolution_clock::now();
        auto engine = gemfinder::EngineFactory::createEngine(model_path);
        auto vec = engine->embedText(text);
        auto end_t = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_t - start_t).count();

        std::cout << "=== GemFinder Embedding Test ===\n";
        std::cout << "Engine:    " << engine->modelName() << "\n";
        std::cout << "Dim:       " << vec.size() << " floats (L2 normalizzato)\n";
        std::cout << "Latency:   " << elapsed_ms << " ms\n";
        std::cout << "Vector:    [";
        for (size_t i = 0; i < std::min<size_t>(8, vec.size()); ++i) {
            std::cout << (i > 0 ? ", " : "") << vec[i];
        }
        std::cout << ", ...]\n";
        return 0;
    }

    std::cerr << "Comando sconosciuto: " << arg1 << "\n";
    printUsage(argv[0]);
    return 1;
}
