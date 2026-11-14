# Piano di Progetto: "GemFinder" – Motore di Ricerca Desktop Multimodale in C/C++

## 1. Visione del Progetto e Obiettivo
**GemFinder** è un sistema di ricerca locale ad alte prestazioni (simile a Spotlight su macOS o Everything su Windows, ma con comprensione semantica profonda) scritto interamente in **C/C++**. 

L'obiettivo è permettere all'utente di inserire nella barra di ricerca **testo, immagini, file audio o una combinazione di essi** e trovare istantaneamente qualsiasi file sul computer semanticamente correlato, indipendentemente dal formato (es. cercare con un file audio di pioggia e trovare foto di temporali, appunti testuali sul meteo o tracce audio simili).

Il cuore semantico del sistema è **EmbeddingGemma 2** di Google, un modello nativamente multimodale (testo, immagini, audio, video, codice) che proietta ogni tipo di dato in uno spazio vettoriale unificato.

---

## 2. Architettura del Sistema

Per garantire ricerche istantanee (< 15 millisecondi) senza appesantire il sistema operativo, il software deve essere diviso in due eseguibili separati che comunicano via **IPC (Inter-Process Communication)**:

1. **`GemFinder-daemon` (Servizio in Background / Demone):**
   * Scansiona il file system all'avvio e monitora in tempo reale le modifiche (creazione, modifica, spostamento, eliminazione).
   * Estrae e pre-processa il contenuto dei file (chunking del testo, ricampionamento audio, ridimensionamento immagini).
   * Esegue l'inferenza tramite **EmbeddingGemma 2** in background (sfruttando GPU/NPU se disponibili, con throttling intelligente).
   * Aggiorna il database dei metadati (SQLite) e il grafo vettoriale (HNSW).

2. **`GemFinder-gui` (Client / Barra di Ricerca):**
   * Applicazione leggerissima in memoria, attivabile tramite scorciatoia da tastiera globale (es. `Alt + Spazio`).
   * Accetta input testuali, drag-and-drop di immagini/audio o registrazione vocale al volo.
   * Invia la query al demone tramite *Unix Domain Sockets* (Linux/macOS) o *Named Pipes* (Windows) e renderizza i risultati in tempo reale.

```
+-----------------------------------------------------------------------+
|                         FILE SYSTEM DEL PC                            |
+-----------------------------------------------------------------------+
                                   | (Eventi: inotify / FSEvents / USN)
                                   v
+-----------------------------------------------------------------------+
|                        GEMFINDER-DAEMON (C++)                          |
|                                                                       |
|  [File Watcher] ---> [Coda Debouncing & Priorità]                     |
|                              |                                        |
|                              v                                        |
|                [Pipeline Pre-processing Media]                        |
|               (Testo / Audio 16kHz / Immagini)                        |
|                              |                                        |
|                              v                                        |
|             [Motore Inferenza: EmbeddingGemma 2]                      |
|                  (llama.cpp / LiteRT-LM)                              |
|                              |                                        |
|              +---------------+---------------+                        |
|              | (Metadati + FTS5)             | (Vettori 256d/768d)    |
|              v                               v                        |
|      +---------------+               +---------------+                |
|      |    SQLite3    | <-----------> | Indice HNSW   |                |
|      |  (Relazionale)|   (chunk_id)  |   (USearch)   |                |
|      +---------------+               +---------------+                |
+-----------------------------------------------------------------------+
                                   ^
                                   | (IPC: Socket / Named Pipe)
                                   v
+-----------------------------------------------------------------------+
|                         GEMFINDER-GUI (C++)                            |
|   [Barra di Ricerca: Testo + Drag&Drop Immagini/Audio + Microfono]    |
+-----------------------------------------------------------------------+
```

---

## 3. Stack Tecnologico C/C++ Consigliato

Per evitare dipendenze pesanti (niente Python o runtime esterni) e mantenere l'eseguibile compatto e veloce:

* **Standard Linguaggio:** **C++20** (per `std::filesystem`, coroutine, thread-pool moderni e `std::span`).
* **Build System:** **CMake** + **vcpkg** o **Conan** per la gestione delle librerie.
* **Motore di Inferenza AI:**
  * **Opzione A (Consigliata):** **`llama.cpp`** integrato come sottomodulo C/C++. Supporta nativamente EmbeddingGemma 2 in formato GGUF (quantizzato Q4_K_M o Q8_0), inclusi gli encoder visivi e audio, con accelerazione Metal (macOS), CUDA/Vulkan (Windows/Linux) e AVX2/NEON (CPU).
  * **Opzione B:** **`LiteRT-LM`** (ex TensorFlow Lite di Google), runtime C++ ufficiale per modelli Gemma on-device.
* **Indice Vettoriale (ANN Search):**
  * **`USearch`** (di Unum): Libreria C++11 *header-only*. Supporta l'algoritmo HNSW, quantizzazione scalare (`f16` o `i8`), cancellazione dinamica dei nodi e **Memory-Mapping (`mmap`)** su disco per caricare milioni di vettori in $0\text{ ms}$ all'avvio.
  * *Alternativa:* **`hnswlib`** (header-only C++) o **`sqlite-vec`** (estensione scritta in C puro per SQLite).
* **Database Metadati e Ricerca Lessicale:**
  * **`SQLite3`** (in modalità WAL - *Write-Ahead Logging* per letture/scritture concorrenti) + estensione nativa **FTS5** per la ricerca testuale esatta (BM25).
* **Librerie di Pre-processing (Single-Header C/C++):**
  * **Immagini:** `stb_image.h` e `stb_image_resize2.h` (leggerissime, in C puro).
  * **Audio:** `miniaudio.h` o `dr_wav.h` / `dr_mp3.h` (per decodificare e ricampionare l'audio a $16\text{ kHz}$ mono). Per formati complessi o estrazione audio da video: **FFmpeg (`libavcodec` / `libavformat`)**.
  * **Documenti:** `PDFium` o `MuPDF` (per estrarre testo e immagini da file PDF), `pugixml` + `miniz` (per leggere `.docx`, `.odt`, `.epub` che sono archivi ZIP contenenti XML).
* **Interfaccia Grafica (GUI):**
  * **`Dear ImGui`** (con backend GLFW/SDL + OpenGL/DirectX/Metal) per un'interfaccia ultraveloce e minimale, oppure **`Qt6`** per un'integrazione desktop più tradizionale.

---

## 4. Specifiche di EmbeddingGemma 2 e Gestione Vettori

EmbeddingGemma 2 (basato su architettura Gemma 3 con encoder multimodali dedicati) presenta caratteristiche specifiche da sfruttare nel codice C++:

1. **Finestra di Contesto:** $8.192\text{ token}$ (equivale a circa 5,5 minuti di audio continuo a $25\text{ token/s}$, oppure fino a 29 immagini in un singolo contesto).
2. **Matryoshka Representation Learning (MRL):** Il modello genera vettori di dimensione $d = 768$, ma grazie all'addestramento MRL è possibile **troncare il vettore alle prime 256 o 128 dimensioni** e ri-normalizzarlo:
   $$\hat{v}_{256} = \frac{v_{1..256}}{\|v_{1..256}\|_2}$$
   * **Vantaggio:** Salvare vettori da $256\text{ float16}$ occupa solo $512\text{ byte}$ per elemento (contro i $3.072\text{ byte}$ di un vettore $768\text{ float32}$), riducendo di 6 volte l'uso della RAM e rendendo la ricerca istantanea anche con 5 milioni di file sul PC.
3. **Formattazione dei Prompt (Fondamentale per la qualità):**
   EmbeddingGemma richiede prefissi specifici per distinguere i documenti indicizzati dalle query dell'utente:
   * **Durante l'indicizzazione di un file sul disco:**
     `title: {nome_file_e_cartella} | text: {contenuto_testo_o_token_media}`
   * **Durante la ricerca nella barra (Query):**
     `task: search result | query: {input_utente_testo_o_media}`
4. **Input Multimodali Combinati (Interleaving):**
   È possibile passare al modello sequenze miste usando i token `<|image|>` e `<|audio|>`, permettendo ricerche ibride (es. trascinare un'immagine nella barra e aggiungere del testo per affinare la ricerca).

---

## 5. Struttura dei Dati: Il Database Ibrido

Il sistema mantiene lo stato del computer sincronizzato attraverso due file principali salvati nella cartella dati dell'applicazione: `metadata.db` (SQLite) e `vectors.usearch` (Indice HNSW).

### Schema SQLite (`metadata.db`)

```sql
-- Tabella principale dei file tracciati sul file system
CREATE TABLE files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    extension TEXT,
    file_size INTEGER,
    mtime INTEGER NOT NULL,          -- Timestamp ultima modifica (Unix epoch)
    xxhash64 INTEGER,                -- Hash veloce per rilevare file rinominati/spostati
    modality TEXT,                   -- 'text', 'image', 'audio', 'video', 'mixed'
    status TEXT DEFAULT 'indexed'    -- 'pending', 'indexed', 'error'
);

-- Tabella dei frammenti (Chunk): un file lungo può generare più vettori
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Corrisponde alla chiave (Key) nell'indice USearch!
    file_id INTEGER NOT NULL,
    chunk_type TEXT,                 -- 'text_paragraph', 'audio_segment', 'image_frame'
    start_offset INTEGER,            -- Riga di testo, secondo dell'audio o pagina del PDF
    end_offset INTEGER,
    preview_text TEXT,               -- Snippet da mostrare nella GUI
    FOREIGN KEY(file_id) REFERENCES files(file_id) ON DELETE CASCADE
);

-- Indice Full-Text Search (FTS5) per ricerca lessicale esatta (BM25)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    filename,
    preview_text,
    content='chunks',
    content_rowid='chunk_id'
);
```

### Sincronizzazione SQLite <-> USearch
Il `chunk_id` (intero a 64 bit) generato da SQLite viene usato direttamente come **ID univoco del vettore** dentro `USearch`.
* **Aggiunta/Modifica:** Quando un file viene modificato, si eliminano i vecchi `chunk_id` sia da SQLite che da `USearch` (`index.remove(chunk_id)`), si ricalcolano gli embedding del file e si inseriscono i nuovi vettori (`index.add(new_chunk_id, vector_ptr)`).
* **Spostamento/Rinomina:** Se cambia solo il percorso del file ma `mtime`, `file_size` e `xxhash64` restano identici, **non si ricalcola l'embedding**: si esegue solo un `UPDATE files SET filepath = ...` in SQLite in meno di $1\text{ ms}$.

---

## 6. Pipeline di Pre-processing per Tipo di File

Per codificare l'intero computer senza saturare la memoria, ogni categoria di file segue una pipeline dedicata:

| Tipo di Contenuto | Estensioni Esempio | Strategia di Pre-processing e Chunking |
| :--- | :--- | :--- |
| **Immagini** | `.jpg`, `.png`, `.webp`, `.heic` | Decodifica con `stb_image`, ridimensionamento alla risoluzione richiesta dall'encoder visivo di EmbeddingGemma 2, generazione di **1 singolo vettore** per immagine (includendo il nome del file e la cartella nel prompt `title:`). |
| **Audio** | `.wav`, `.mp3`, `.flac`, `.m4a` | Decodifica con `miniaudio`, conversione in **PCM 16 kHz Mono float32**. Se l'audio supera i 60 secondi, suddivisione in finestre di **30 secondi** con 5 secondi di overlap. Ogni segmento genera un vettore con `start_offset` = secondo di inizio. |
| **Testo & Codice** | `.txt`, `.md`, `.cpp`, `.py`, `.json` | Suddivisione semantica (per paragrafi o funzioni) in blocchi da **512–1024 token** con overlap di 64 token. Ogni blocco diventa un chunk. |
| **Documenti Ricchi** | `.pdf`, `.docx` | Estrazione del testo per pagina/sezione + estrazione delle immagini incorporate sopra una certa dimensione (es. $> 150 \times 150\text{ px}$). Sia i paragrafi che le immagini estratte vengono indicizzati puntando allo stesso `file_id` e al numero di pagina (`start_offset`). |
| **Video (Opzionale)** | `.mp4`, `.mkv`, `.mov` | Estrazione della traccia audio (chunk da 30s) + estrazione di 1 keyframe visivo ogni 10-15 secondi (o al cambio scena) tramite `libavcodec`. Permette di trovare il punto esatto di un video cercando sia ciò che si vede sia ciò che si sente. |
| **Altri File / Binari** | `.zip`, `.exe`, `.blend` | Nessuna lettura del contenuto binario; si indicizza solo il percorso, il nome del file e i metadati di sistema come testo (`title: ...`). |

---

## 7. Monitoraggio del File System in Tempo Reale

Per fare in modo che l'indice sia sempre aggiornato senza dover scansionare continuamente il disco, il demone utilizza le API native del kernel:

* **Linux:** **`inotify`** (o **`fanotify`** per il monitoraggio dell'intero mount point).
* **macOS:** **`FSEvents` API** (estremamente efficiente, permette anche di interrogare gli eventi avvenuti mentre l'app era chiusa tramite `FSEventStreamCreate` con un *Event ID* storico).
* **Windows:** **`ReadDirectoryChangesW`** per il monitoraggio live + lettura diretta della **MFT (Master File Table) e dell'USN Change Journal** su volumi NTFS per sincronizzare istantaneamente le modifiche avvenute a computer spento o demone chiuso.

### Gestione degli Eventi: Coda di Debouncing e Filtri
1. **Filtro Esclusioni Immediato:** Prima ancora di mettere un evento in coda, scartare percorsi inutili tramite regole stile `.gitignore` (es. `/proc`, `/sys`, `C:\Windows`, `node_modules/`, `.git/`, `build/`, `.cache/`, file temporanei `.tmp` o `.swp`).
2. **Debouncer (Coalescing Engine):** Se un utente sta salvando un file o compilando un progetto, il sistema operativo può generare decine di eventi `MODIFY` in pochi millisecondi per lo stesso file. Il demone inserisce i percorsi modificati in una `std::unordered_map<std::string, TimePoint>` e avvia l'indicizzazione solo quando il file non subisce ulteriori modifiche per **2 secondi**.
3. **Coda a Priorità:**
   * *Priorità Alta:* File appena creati o modificati dall'utente nelle cartelle `Desktop`, `Documenti`, `Download`.
   * *Priorità Bassa:* Scansione iniziale di archivi storici o cartelle secondarie.

---

## 8. Il Motore di Query: Ricerca Ibrida Istantanea

Quando l'utente inserisce un input nella barra di ricerca (testo, audio, immagine o combinazione), il flusso esegue i seguenti passaggi in meno di **15–30 ms**:

1. **Embedding della Query:**
   Il contenuto inserito nella barra viene passato a EmbeddingGemma 2 (che rimane sempre pre-caricato nella RAM/VRAM del demone). Viene estratto il vettore $q \in \mathbb{R}^{256}$.
2. **Ricerca Vettoriale (HNSW):**
   Si interroga l'indice `USearch` in memoria per ottenere i primi $K = 50$ `chunk_id` più vicini secondo la *Cosine Similarity*:
   $$\text{sim}(q, v) = \frac{q \cdot v}{\|q\|_2 \|v\|_2}$$
3. **Ricerca Lessicale Parallela (Solo se l'input contiene testo):**
   Contemporaneamente, si esegue una query sulla tabella `chunks_fts` di SQLite usando l'algoritmo **BM25** per catturare corrispondenze esatte di nomi file, codici o sigle.
4. **Fusione dei Risultati (Reciprocal Rank Fusion - RRF):**
   I risultati semantici (vettoriali) e quelli lessicali (BM25) vengono combinati con la formula RRF per produrre la classifica finale:
   $$\text{RRF\_Score}(d) = \frac{1}{k + \text{rank}_{\text{vector}}(d)} + \frac{1}{k + \text{rank}_{\text{bm25}}(d)}$$
   *(dove $k = 60$ è una costante standard di smoothing)*.
5. **Deduplicazione per File e Risposta:**
   Se più chunk appartengono allo stesso file, vengono raggruppati mostrando il file una sola volta con il punteggio migliore e l'indicazione precisa del punto trovato (es. *"Trovato a pagina 4"* o *"Trovato al minuto 02:15"*).

---

## 9. Roadmap di Sviluppo Step-by-Step

### Fase 1: Proof of Concept (CLI e Inferenza)
* [x] Configurare il progetto CMake in C++20.
* [ ] Integrare `llama.cpp` (o `LiteRT-LM`) e scaricare i pesi quantizzati di **EmbeddingGemma 2**.
* [ ] Scrivere un tool a riga di comando che prende un testo, un'immagine `.jpg` (tramite `stb_image`) e un file `.wav` (tramite `miniaudio`), calcola gli embedding a 256 dimensioni e stampa la matrice di similarità coseno tra i tre elementi.

### Fase 2: Storage Layer (SQLite + USearch)
* [x] Integrare `sqlite3.c` e `usearch/index_dense.hpp` nel progetto.
* [x] Implementare la classe `IndexManager` in C++ che gestisce le transazioni: inserimento di un file, salvataggio dei chunk in SQLite, inserimento dei vettori in USearch e salvataggio/caricamento tramite `mmap`.
* [x] Implementare l'hashing veloce dei file con `xxHash` per distinguere le modifiche reali dai semplici spostamenti di file.

### Fase 3: Crawler Iniziale e Pre-processori
* [ ] Costruire lo scanner ricorsivo (`std::filesystem::recursive_directory_iterator`) con il filtro di esclusione delle directory di sistema.
* [ ] Implementare i moduli di chunking per testo, audio (segmenti da 30s) e immagini.
* [ ] Creare un *Thread Pool* per pre-processare i file su CPU mentre un thread dedicato alimenta la GPU/CPU con batch di inferenza per EmbeddingGemma 2.

### Fase 4: File System Watcher e Demone Background
* [ ] Implementare il wrapper multipiattaforma per `inotify` / `FSEvents` / `ReadDirectoryChangesW` (oppure integrare una libreria C++ matura come **`efsw`** - *Entropia File System Watcher*).
* [ ] Costruire la coda con *debouncing* e il sistema di pausa automatica quando il PC è a batteria o la CPU supera l'80% di carico utente.
* [ ] Esporre il server IPC locale (Unix Socket / Named Pipe).

### Fase 5: Barra di Ricerca (GUI) ed Esperienza Utente
* [ ] Creare l'interfaccia stile Spotlight in `Dear ImGui` o `Qt6`.
* [ ] Implementare il Drag & Drop di file (immagini/audio) direttamente dentro la barra di ricerca e il pulsante "Registra Audio" per cercare tramite voce o suoni ambientali.
* [ ] Visualizzare le anteprime dei risultati con evidenziazione del testo o mini-player per saltare direttamente al secondo esatto del file audio/video trovato.

---

## 10. Riferimenti e Progetti Open Source da Studiare

* **Repository Ufficiali e Modelli:**
  * [Google EmbeddingGemma su HuggingFace](https://huggingface.co/google) – Pesi ufficiali e documentazione sui prompt.
  * [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) – Studiare la cartella `examples/embedding` e il supporto multimodale per vedere come vengono passati i buffer float di immagini e audio in C++.
  * [google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) – Runtime C++ di Google per modelli Gemma.
* **Librerie C/C++ Chiave da Includere:**
  * [unum-cloud/usearch](https://github.com/unum-cloud/usearch) – Motore HNSW single-header velocissimo con supporto `mmap`.
  * [Sackers/efsw](https://github.com/SpartanJ/efsw) – *Entropia File System Watcher*, libreria C++ multipiattaforma senza dipendenze per monitorare directory su Linux, macOS e Windows.
  * [mackron/miniaudio](https://github.com/mackron/miniaudio) – Libreria C single-file per decodifica, ricampionamento audio a 16 kHz e cattura da microfono.
  * [Cyan4973/xxHash](https://github.com/Cyan4973/xxHash) – Algoritmo di hashing in C capace di lavorare alla velocità della RAM per rilevare istantaneamente i file modificati o spostati.
* **Ispirazione Progettuale:**
  * [lattifai/awesome-EmbeddingGemma](https://github.com/lattifai/awesome-EmbeddingGemma) – Lista aggiornata di progetti che usano EmbeddingGemma 2 (come *DigUp* e *MemPalace*).