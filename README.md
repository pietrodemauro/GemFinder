# GemFinder

[![C++20](https://img.shields.io/badge/Language-C%2B%2B20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux-orange.svg)](https://kernel.org)
[![Build](https://img.shields.io/badge/Build-CMake-brightgreen.svg)](https://cmake.org)

**GemFinder** is an ultra-fast, offline desktop multimodal semantic search engine written in modern C++20. It indexes and searches across local documents, source code, images, audio files, and PDFs using local transformer models (EmbeddingGemma 2 via `llama.cpp` and `mtmd`), hybrid retrieval (Reciprocal Rank Fusion over SQLite3 FTS5 and USearch HNSW), and a responsive Spotlight/Raycast-style Qt 6 interface.

---

## Key Features

- **Multimodal by Design**:
  - **Text & Source Code**: Syntax-aware paragraph segmentation and MRL embedding vectors.
  - **PDF Documents with Native & OCR Support**: Fast-path native text extraction with automatic fallback to high-resolution OCR (Poppler C++ + Tesseract OCR) for scanned documents.
  - **Images**: Vision Transformer encoding (`mmproj`) for visual similarity search and text-to-image semantic matching.
  - **Audio**: 16 kHz Mono normalization and segment-level embeddings for acoustic and semantic search.
- **Hardware-Accelerated Inference**:
  - Full Vulkan GPU acceleration enabled by default via `llama.cpp` and `ggml`.
  - Configurable CPU thread pooling (capped at 50% CPU cores by default to prevent desktop freezes).
- **Hybrid Search Engine**:
  - **SQLite3 WAL + FTS5**: Ultra-low latency lexical BM25 retrieval.
  - **USearch HNSW**: Vector indexing with 256-dimensional $L_2$-normalized Matryoshka representations.
  - **Reciprocal Rank Fusion (RRF)**: Merges keyword and vector ranks into a single unified relevance score.
- **Background Daemon (`inotify`)**:
  - Real-time filesystem watcher with event debouncing.
  - Instantaneous IPC communication over UNIX domain sockets (`/tmp/gemfinder.sock`).
- **Spotlight/Raycast GUI**:
  - Real-time keyboard-first search.
  - Results grouped by modality (**Documents & Text**, **Images**, **Audio**).
  - Directory indexing and unindexing directly from the UI.
  - Drag-and-drop support for image and audio queries.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │          gemfinder-gui (Qt 6)           │
                    │   [Documents]   [Images]   [Audio]      │
                    └────────────────────┬────────────────────┘
                                         │ UNIX Domain Socket (/tmp/gemfinder.sock)
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           gemfinder-daemon                                  │
│  ┌──────────────────────┐                     ┌──────────────────────────┐  │
│  │ inotify FileWatcher  │                     │   EmbeddingGemma 2       │  │
│  │ & Debounce Queue     │                     │ (llama.cpp Vulkan / CPU) │  │
│  └──────────┬───────────┘                     └────────────┬─────────────┘  │
│             │                                              │                │
│             ▼                                              ▼                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │     StorageManager: SQLite3 FTS5 (BM25) + USearch HNSW (Vectors)      │  │
│  │     Hybrid Retrieval via Reciprocal Rank Fusion (RRF)                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation & Requirements

### Dependencies (Arch Linux / Fedora / Ubuntu)

#### Arch Linux:
```bash
sudo pacman -S base-devel cmake qt6-base sqlite poppler tesseract tesseract-data-eng vulkan-headers
```

#### Ubuntu / Debian (22.04+):
```bash
sudo apt install build-essential cmake qt6-base-dev libsqlite3-dev libpoppler-cpp-dev libtesseract-dev tesseract-ocr libvulkan-dev
```

### Build from Source

```bash
# Clone the repository
git clone https://github.com/gemfinder/gemfinder.git
cd gemfinder

# Configure and compile
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Run test suite:
```bash
ctest --test-dir build --output-on-failure
```

Install system binaries:
```bash
sudo cmake --install build
```

---

## Quickstart

### 1. Download Model Weights
Place the quantized GGUF weights into `models/`:
```bash
mkdir -p models
# Place embeddinggemma-2.gguf and mmproj.gguf in models/
```

### 2. Index a Directory
```bash
gemfinder-cli index ~/Documents --model models/embeddinggemma-2.gguf
```

### 3. Run the Background Daemon
```bash
gemfinder-daemon --watch ~/Documents --watch ~/Pictures --model models/embeddinggemma-2.gguf
```

### 4. Launch the GUI
```bash
gemfinder-gui
```

Global hotkey shortcut toggle:
```bash
gemfinder-toggle
```

---

## CLI Reference

| Command | Description |
|---|---|
| `gemfinder-cli index <dir>` | Indexes all supported files in directory |
| `gemfinder-cli unindex <dir>` | Removes directory from index |
| `gemfinder-cli search <query>` | Hybrid semantic text search |
| `gemfinder-cli search --image <file>` | Visual similarity search using query image |
| `gemfinder-cli search --audio <file>` | Acoustic similarity search using query audio |
| `gemfinder-cli list` | Lists all indexed files and status |
| `gemfinder-cli stats` | Shows index metrics, vector count, and database size |
| `gemfinder-cli inspect <file>` | Preprocesses file and prints chunking details |
| `gemfinder-cli embed <text>` | Computes raw embedding vector and latency |

For full documentation, see [User Guide](docs/USER_GUIDE.md).

---

## License & Author

- **Author**: Pietro
- **License**: GNU General Public License v3.0 (GPL-3.0). See [LICENSE](LICENSE) for the full text.
