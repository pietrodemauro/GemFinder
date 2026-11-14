# Piano Esecutivo di Sviluppo: GemFinder

Questo documento definisce la roadmap tecnica operativa e incrementale per l'implementazione di **GemFinder**, motore di ricerca multimodale semantico desktop in C++20.

---

## 1. Obiettivo e Vincoli del Sistema

* **Target OS / Architettura:** Linux x86_64 (CachyOS / Arch Linux), estendibile a Windows/macOS.
* **Linguaggio e Standard:** C++20 (`std::filesystem`, coroutine, `std::span`, jthread).
* **Toolchain:** CMake 4.x + GCC 16.x / Clang.
* **Accelerazione Hardware:** Vulkan API per GPU AMD Radeon RX 6700 XT (12GB VRAM), fallback CPU con istruzioni AVX2 per AMD Ryzen 7 5800X.
* **Modello:** EmbeddingGemma 2 (quantizzato GGUF) con estrazione Matryoshka ($d=256$, float32/float16, normalizzato $L_2$).
* **Tempo di risposta target per query:** $< 25\text{ ms}$ (ibrido semantico + lessicale).

---

## 2. Struttura del Repository

```text
GemFinder/
├── CMakeLists.txt
├── .gitignore
├── docs/
│   ├── architettura_e_piano_di_sviluppo_multimodal_semantic_finder.md
│   └── piano_esecutivo_sviluppo.md
├── third_party/             # Dipendenze header-only e submoduli
│   ├── usearch/             # HNSW vector engine (Unum)
│   ├── xxhash/              # xxHash per hashing ultra-rapido dei file
│   ├── stb/                 # stb_image.h, stb_image_resize2.h
│   └── miniaudio/           # miniaudio.h per decodifica e resampling audio
├── src/
│   ├── common/              # Tipi condivisi (Types.hpp), Logging, Config, ErrorHandling
│   ├── storage/             # SQLite3 (metadata + FTS5) e USearch IndexManager
│   ├── preprocessor/        # Pipeline per testo, immagini e audio (PCM 16kHz mono)
│   ├── inference/           # Wrapper GGUF/llama.cpp con Vulkan + normalizzazione MRL
│   ├── engine/              # HybridSearchEngine (orchestratore query, ranking RRF)
│   ├── daemon/              # Background service, efsw file watcher, Unix domain socket IPC
│   ├── cli/                 # gemfinder-cli: utility da riga di comando per indicizzare e cercare
│   └── gui/                 # gemfinder-gui: client grafico minimale (Dear ImGui)
└── tests/                   # Test unitari e benchmark prestazionali
```

---

## 3. Fasi Operative Dettagliate

### Step 0: Setup del Progetto e Build System (Fondamenta) [COMPLETATO]
* **Obiettivo:** Creare lo scheletro modulare del progetto compilabile con CMake in C++20.
* **Stato:** ✅ Completato il 14/11/2026.
* **Risultati:**
  - Repository git inizializzato (branch `main`), configurato `.gitignore`.
  - Struttura directory (`src/common`, `storage`, `preprocessor`, `inference`, `engine`, `daemon`, `cli`, `gui`).
  - Standard C++20 con CMake, flags `-Wall -Wextra -Wpedantic -O3 -march=native`.
  - Moduli iniziali: `Types.hpp`, `Hasher.hpp` (xxHash64 con streaming), `Logger.hpp`, `Version.hpp`.
  - CLI iniziale `gemfinder-cli` compilata e funzionante (`--info`, `--hash`, `--version`).

### Step 1: Storage Layer & Ricerca Ibrida (SQLite3 + USearch) [COMPLETATO]
* **Obiettivo:** Persistenza su disco dei metadati dei file, frammentazione (chunking) e indicizzazione vettoriale scalabile.
* **Stato:** ✅ Completato il 14/11/2026.
* **Risultati:**
  - `Database`: SQLite3 in modalità WAL, tabelle `files`, `chunks`, e tabella virtuale `chunks_fts` (FTS5 BM25 con snippet e unicode61).
  - `VectorStore`: integrazione HNSW `USearch` (unum-cloud/usearch v2.26) a 256 dimensioni (metrica Cosine / float32) con persistenza e zero-latency `mmap`.
  - `StorageManager`: sincronizzazione atomica tra SQLite e USearch (`chunk_id <-> vector_key`), aggiornamento incrementale e algoritmo di fusione **RRF (Reciprocal Rank Fusion)** per ricerca ibrida.
  - Test suite unitaria `tests/test_storage.cpp` (FTS5, VectorStore, RRF Hybrid Search) integrata in CTest e superata al 100%.

### Step 2: Pipeline di Pre-processing Multimodale [COMPLETATO]
* **Obiettivo:** Trasformare file su disco in strutture tokenizzabili o tensori pronti per l'inferenza.
* **Stato:** ✅ Completato il 14/11/2026.
* **Risultati:**
  - `TextPreprocessor`: segmentazione semantica intelligente su confini di paragrafo (`\n\n`), frase o parole, tracciamento righe (`start_line`/`end_line`), generazione anteprime e prompt formatting Gemma (`title: ... | text: ...` e `task: search result | query: ...`).
  - `ImagePreprocessor`: decodifica immagini tramite `stb_image.h` (JPEG, PNG, BMP, WebP), ridimensionamento bilineare continuo a $224 \times 224$ e normalizzazione canali RGB in float `[0.0, 1.0]`.
  - `AudioPreprocessor`: decodifica multi-formato tramite `miniaudio.h` (WAV, MP3, FLAC), resampling automatico a **PCM 16 kHz Mono float32**, slicing in finestre da 30s con overlap di 5s.
  - `MediaPreprocessor`: router automatico delle estensioni con calcolo hash `xxhash64` e generazione del payload completo `ProcessedPayload`.
  - Integrazione CLI (`gemfinder-cli --inspect <file>`) e test unitari `test_preprocessor` con 100% test superati in CTest.

### Step 3: Inference Engine (EmbeddingGemma 2 via llama.cpp / Vulkan) [COMPLETATO]
* **Obiettivo:** Generazione locale ad alta efficienza di vettori densi multimodali.
* **Stato:** ✅ Completato il 14/11/2026.
* **Risultati:**
  - `InferenceEngine`: interfaccia C++ astratta unificata per inferenza multimodale (`embedText`, `embedImage`, `embedAudio`).
  - `LlamaEngine`: integrazione nativa con `llama.cpp` (sottomodulo compilato con supporto multithread OpenMP e AVX2), context con `embeddings = true`, mean pooling, estrazione token, decodifica ed estrazione vettori.
  - `applyMrlAndNormalize`: algoritmo di Matryoshka Representation Learning (MRL) per troncamento da dimensione raw (es. 768) a $256$ float con normalizzazione euclidea $L_2$ ($\|\hat{v}\|_2 = 1.0$).
  - `DeterministicMockEngine`: embedder deterministico per test e sviluppo locale senza dipendenza forzata da download multi-GB di modelli.
  - `EngineFactory`: caricamento trasparente con fallback controllato.
  - Test unitario `test_inference` (MRL, Cosine Similarity, Factory) superato al 100% in CTest.
  - CLI `gemfinder-cli --embed "<testo>"` funzionante con latenza sub-millisecondo (< 0.1 ms).

### Step 4: Prototipo CLI End-to-End (`gemfinder-cli`) [COMPLETATO]
* **Obiettivo:** Testare l'intero flusso di indicizzazione e interrogazione da terminale.
* **Stato:** ✅ Completato il 14/11/2026.
* **Risultati:**
  - `Crawler`: scansione ricorsiva ad alte prestazioni con filtri di esclusione directory (`.git`, `node_modules`, `build`, `.cache`, `third_party`, ecc.) e file nascosti.
  - `SearchEngine`: orchestrazione completa tra Crawler, MediaPreprocessor, InferenceEngine e StorageManager.
  - Scansione incrementale intelligente: verifica di `mtime`, `file_size` e `xxhash64` per saltare istantaneamente file non modificati a costo zero.
  - Ricerca ibrida multimodale: testo, immagini e audio con ranking RRF combinato (HNSW + FTS5 BM25) e latenze di query misurate **< 1 ms** su oltre 115.000 vettori.
  - Comandi CLI implementati: `index`, `search` (con `--image` e `--audio`), `stats`, `inspect`, `embed`, `hash`.
  - Suite unitaria `test_engine` integrata in CTest con 100% di test superati.

### Step 5: Demone di Background e File Watcher [COMPLETATO]
* **Obiettivo:** Monitoraggio continuo e silenzioso del disco senza rallentare il sistema.
* **Stato:** ✅ Completato il 14/11/2026.
* **Risultati:**
  - `FileWatcher`: monitoraggio in tempo reale del kernel Linux tramite API `inotify` non-bloccante (`IN_CREATE`, `IN_MODIFY`, `IN_DELETE`, `IN_MOVED_FROM`, `IN_MOVED_TO`, `IN_CLOSE_WRITE`) con aggiunta dinamica ricorsiva di nuove directory e filtri di esclusione automatici.
  - `Debouncer`: raggruppamento (*coalescing*) intelligente degli eventi su mappa concorrente con ritardo configurabile (default 2 secondi) per assorbire salvataggi multipli ed evitare accessi disco ridondanti.
  - `IpcServer` e `IpcClient`: server e client su **Unix Domain Socket** (`/tmp/gemfinder.sock`) ad altissima velocità (latenze roundtrip di query < 0.3 ms) con protocollo tabulato a latenza zero.
  - Eseguibile `gemfinder-daemon`: demone di background che mantiene l'indice vettoriale e i modelli caldi in memoria, gestisce il monitoraggio live delle cartelle e risponde alle query IPC con spegnimento pulito (`SIGINT`/`SIGTERM`).
  - Integrazione trasparente nel `gemfinder-cli`: le ricerche interrogano automaticamente il demone in esecuzione se presente per risposte istantanee.
  - Suite unitaria `test_daemon` integrata in CTest con 100% test superati.

### Step 6: Client GUI Desktop (Spotlight-like) [PROSSIMO PASSO]
* **Obiettivo:** Interfaccia utente scattante (< 50ms) attivabile via scorciatoia da tastiera globale.
* **Azioni:**
  1. Creazione finestra flottante minimale (Dear ImGui con backend GLFW / OpenGL / Vulkan).
  2. Barra di ricerca unificata (input testuale, drag-and-drop di immagini/audio, pulsante cattura microfono).
  3. Rendering dei risultati con evidenziazione del match e indicazione precisa della pagina o del minuto temporale.
