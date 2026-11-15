# GemFinder User Guide

This guide covers how to set up, index, and use **GemFinder** with **EmbeddingGemma 2** (quantized GGUF format) for sub-millisecond multimodal semantic searches across text documents, source code, images, audio, and PDF files.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites and Model Downloads](#2-prerequisites-and-model-downloads)
3. [Building and Installation](#3-building-and-installation)
4. [Step 1: Initial Directory Indexing (CLI)](#4-step-1-initial-directory-indexing-cli)
5. [Step 2: Running the Background Daemon (`gemfinder-daemon`)](#5-step-2-running-the-background-daemon-gemfinder-daemon)
6. [Step 3: Desktop Spotlight GUI (`gemfinder-gui`)](#6-step-3-desktop-spotlight-gui-gemfinder-gui)
7. [Step 4: Global Hotkey Integration](#7-step-4-global-hotkey-integration)
8. [Step 5: Advanced CLI Operations](#8-step-5-advanced-cli-operations)
9. [Troubleshooting & FAQ](#9-troubleshooting--faq)

---

## 1. Architecture Overview

GemFinder combines:
- **Inference Engine (`llama.cpp` + `mtmd`):** Runs EmbeddingGemma 2 with GPU acceleration (Vulkan default on Linux) extracting Matryoshka Representation Learning (MRL) embeddings truncated to **256 dimensions** with $L_2$ normalization.
- **Hybrid Storage Layer:** SQLite3 WAL mode with **FTS5 (BM25)** for lexical search and **USearch (HNSW)** for approximate nearest neighbor vector search.
- **Reciprocal Rank Fusion (RRF):** Merges vector and keyword scores via:
  $$RRF(d) = \frac{1}{60 + \text{rank}_{\text{vector}}} + \frac{1}{60 + \text{rank}_{\text{bm25}}}$$
- **Spotlight/Raycast GUI (Qt 6):** Rapid-access desktop window categorizing results into **Text**, **Images**, and **Audio**.

---

## 2. Prerequisites and Model Downloads

### System Dependencies
- C++20 compliant compiler (GCC 12+, Clang 15+)
- CMake 3.20+
- Qt 6 (Widgets)
- SQLite 3 (with FTS5)
- Poppler C++ (`poppler-cpp`)
- Tesseract OCR (`tesseract` and trained data)
- Vulkan SDK and GPU driver (for hardware acceleration)

### Downloading Models
Place the GGUF weights inside `models/`:
```bash
mkdir -p models
# Download EmbeddingGemma 2 embedding model and mmproj vision projector
# Place embeddinggemma-2.gguf and mmproj.gguf in models/
```

Tesseract models (`eng.traineddata`, `ita.traineddata`) are automatically read from `models/tessdata/` or system paths `/usr/share/tessdata`.

---

## 3. Building and Installation

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Run tests to ensure everything is operational:
```bash
ctest --test-dir build --output-on-failure
```

To install binaries into system path (`/usr/local/bin`):
```bash
sudo cmake --install build
```

---

## 4. Step 1: Initial Directory Indexing (CLI)

Index a target directory using `gemfinder-cli`:
```bash
./build/gemfinder-cli index ~/Documents --model models/embeddinggemma-2.gguf --db ~/.local/share/gemfinder/gemfinder_metadata.db
```

Parameters:
- `--model <path.gguf>`: Path to GGUF model.
- `--threads, -t <N>`: Number of CPU worker threads (default: 50% of available CPU cores).
- `--db <path>`: SQLite database destination.
- `--force`: Reindex all files even if timestamps/hashes match.

---

## 5. Step 2: Running the Background Daemon (`gemfinder-daemon`)

The daemon monitors directories for modifications, updates the index via inotify debouncing, and answers IPC search queries via UNIX domain socket (`/tmp/gemfinder.sock`):

```bash
./build/gemfinder-daemon \
  --db ~/.local/share/gemfinder/gemfinder_metadata.db \
  --model models/embeddinggemma-2.gguf \
  --watch ~/Documents \
  --watch ~/Downloads \
  --watch ~/Pictures
```

### Systemd User Service Setup
```bash
mkdir -p ~/.config/systemd/user
cp systemd/gemfinder-daemon.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gemfinder-daemon.service
```

Check status:
```bash
systemctl --user status gemfinder-daemon.service
```

---

## 6. Step 3: Desktop Spotlight GUI (`gemfinder-gui`)

Launch the search UI:
```bash
./build/gemfinder-gui
```

Features:
- Type natural language queries in real time.
- Filter by tabs: **All (Grouped)**, **Text**, **Images**, **Audio**.
- Click **Index Directory...** or **Remove Directory...** to manage indexed locations directly.
- Drag & drop images or audio clips directly into the search window to find semantically similar files.
- Double click any result card to launch the file in its default system application.

---

## 7. Step 4: Global Hotkey Integration

To toggle GemFinder with a global keyboard shortcut (e.g., `Super+Space` or `Alt+Space`):

### GNOME / KDE / XFCE
Assign a custom shortcut running:
```bash
gemfinder-toggle
```
(or `scripts/gemfinder-toggle.sh`).

### Hyprland (`hyprland.conf`)
```ini
bind = SUPER, SPACE, exec, /usr/local/bin/gemfinder-toggle
```

### Sway / i3 (`config`)
```ini
bindsym $mod+space exec /usr/local/bin/gemfinder-toggle
```

---

## 8. Step 5: Advanced CLI Operations

### Semantic Text Search
```bash
./build/gemfinder-cli search "lease contract agreement" --limit 5
```

### Image Similarity Search
```bash
./build/gemfinder-cli search --image sunset.jpg --limit 5
```

### Audio Similarity Search
```bash
./build/gemfinder-cli search --audio sample.wav --limit 5
```

### Unindex a Directory
```bash
./build/gemfinder-cli unindex ~/Downloads
```

### Inspect File Preprocessing & Chunks
```bash
./build/gemfinder-cli inspect document.pdf
```

### List Indexed Files
```bash
./build/gemfinder-cli list
```

---

## 9. Troubleshooting & FAQ

- **GPU Acceleration:** GemFinder compiles with Vulkan backend by default (`GGML_VULKAN ON`). Ensure `vulkan-icd-loader` and graphics drivers are installed.
- **CPU Throttling:** By default, GemFinder allocates 50% of available CPU cores to maintain UI responsiveness. Use `--threads <N>` or `-t <N>` to adjust.
- **PDF Scanned Documents:** If a PDF page contains no embedded text, GemFinder automatically invokes Tesseract OCR at 150 DPI.
