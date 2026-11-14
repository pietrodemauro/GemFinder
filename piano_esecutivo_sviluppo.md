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

### Step 2: Pipeline di Pre-processing Multimodale [PROSSIMO PASSO]
* **Obiettivo:** Trasformare file su disco in strutture tokenizzabili o tensori pronti per l'inferenza.
* **Azioni:**
  1. **Modulo Testo/Codice (`TextPreprocessor`):**
     - Riconoscimento codifica UTF-8.
     - Segmentazione per paragrafi/linee con finestra scorrevole (es. 512 token/caratteri con overlap).
  2. **Modulo Immagini (`ImagePreprocessor`):**
     - Integrazione `stb_image.h` per caricamento JPEG/PNG/WebP.
     - Ridimensionamento bilineare/bicubico (`stb_image_resize2.h`) e normalizzazione canali RGB `[0.0, 1.0]`.
  3. **Modulo Audio (`AudioPreprocessor`):**
     - Integrazione `miniaudio.h` (e `libswresample` per formati complessi).
     - Decodifica e ricampionamento forzato a **PCM 16 kHz mono float32**.
     - Chunking dell'audio in finestre temporali da 30s con 5s di overlap per tracce lunghe.
  4. **Modulo Hash rapido (`FileHasher`):**
     - Calcolo `xxhash64` per verificare se un file modificato è solo stato spostato/rinominato senza alterare il contenuto.

### Step 3: Inference Engine (EmbeddingGemma 2 via llama.cpp / Vulkan)
* **Obiettivo:** Generazione locale ad alta efficienza di vettori densi multimodali.
* **Azioni:**
  1. Integrazione di `llama.cpp` come dipendenza CMake con backend Vulkan abilitato per la GPU Radeon RX 6700 XT.
  2. Configurazione del loader per modelli multimodali GGUF (EmbeddingGemma 2).
  3. Implementazione della formattazione prompt specifica di Gemma:
     - Documenti: `title: {filepath} | text: {chunk_content}`
     - Query: `task: search result | query: {user_input}`
  4. Estrazione del vettore a 768 dimensioni e applicazione **MRL**:
     - Troncamento a 256 dimensioni: $\hat{v}_{256} = v_{1..256}$.
     - Normalizzazione euclidea: $\frac{\hat{v}}{\|\hat{v}\|_2}$.
  5. Test benchmark: misurazione dei millisecondi per embedding su CPU vs GPU Vulkan.

### Step 4: Prototipo CLI End-to-End (`gemfinder-cli`)
* **Obiettivo:** Testare l'intero flusso di indicizzazione e interrogazione da terminale.
* **Azioni:**
  1. Comando `gemfinder-cli index --path <dir>`:
     - Scansione ricorsiva della directory con filtri di esclusione (file nascosti, cartelle di sistema).
     - Pipeline completa: Preprocessing -> Inference -> StorageManager.
  2. Comando `gemfinder-cli search --query "testo"`:
     - Embedding della query -> HNSW search + FTS5 BM25 search -> RRF fusion -> visualizzazione risultati con anteprime ed estratti.
  3. Comando `gemfinder-cli search --image <path>` o `--audio <path>`:
     - Ricerca cross-modale tramite file multimediale.

### Step 5: Demone di Background e File Watcher
* **Obiettivo:** Monitoraggio continuo e silenzioso del disco senza rallentare il sistema.
* **Azioni:**
  1. Integrazione del watcher filesystem (API `inotify` tramite libreria `efsw`).
  2. Implementazione del **Debouncing & Priority Queue**:
     - Coda di eventi con ritardo di 2 secondi per evitare scritture concorrenti durante salvataggi multipli o compilazioni.
     - Esclusione immediata dei path in `.gitignore` o cartelle cache (`node_modules`, `.cache`, `target`, ecc.).
  3. Server IPC basato su **Unix Domain Socket** (`/tmp/gemfinder.sock` o in `~/.local/state/gemfinder/`):
     - Protocollo binario/JSON leggero per richieste di query, stato del demone e trigger di scansione.

### Step 6: Client GUI Desktop (Spotlight-like)
* **Obiettivo:** Interfaccia utente scattante (< 50ms) attivabile via scorciatoia da tastiera globale.
* **Azioni:**
  1. Creazione finestra flottante minimale (Dear ImGui con backend GLFW / OpenGL / Vulkan).
  2. Barra di ricerca unificata (input testuale, drag-and-drop di immagini/audio, pulsante cattura microfono).
  3. Rendering dei risultati con evidenziazione del match e indicazione precisa della pagina o del minuto temporale.
