#include "common/Version.hpp"
#include "common/Logger.hpp"
#include "common/Hasher.hpp"
#include "common/Types.hpp"
#include <sqlite3.h>
#include <iostream>
#include <string_view>
#include <filesystem>

void printUsage(std::string_view progName) {
    std::cout << progName << " - " << gemfinder::kProjectDescription << "\n\n"
              << "Uso:\n"
              << "  " << progName << " --version           Mostra la versione del software\n"
              << "  " << progName << " --help              Mostra questo messaggio di aiuto\n"
              << "  " << progName << " --info              Mostra le librerie e lo stato del sistema\n"
              << "  " << progName << " --hash <file>       Calcola l'hash xxHash64 di un file\n";
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

    std::cerr << "Comando sconosciuto: " << arg1 << "\n";
    printUsage(argv[0]);
    return 1;
}
