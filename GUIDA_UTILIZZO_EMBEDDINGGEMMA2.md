# Guida Completa all'Utilizzo di GemFinder con EmbeddingGemma 2

Questa guida spiega passo dopo passo come configurare, indicizzare e utilizzare **GemFinder** sfruttando il modello di inferenza multimodale **EmbeddingGemma 2** (in formato quantizzato GGUF) per ricerche semantiche istantanee su file di testo, immagini e audio.

---

## Indice
1. [Panoramica dell'Architettura](#1-panoramica-dellarchitettura)
2. [Requisiti e Download di EmbeddingGemma 2 GGUF](#2-requisiti-e-download-di-embeddinggemma-2-gguf)
3. [Compilazione e Installazione](#3-compilazione-e-installazione)
4. [Passo 1: Indicizzazione Iniziale dei Documenti (CLI)](#4-passo-1-indicizzazione-iniziale-dei-documenti-cli)
5. [Passo 2: Avvio del Demone di Background (`gemfinder-daemon`)](#5-passo-2-avvio-del-demone-di-background-gemfinder-daemon)
6. [Passo 3: Ricerca Desktop con GUI Spotlight (`gemfinder-gui`)](#6-passo-3-ricerca-desktop-con-gui-spotlight-gemfinder-gui)
7. [Passo 4: Scorciatoia Globale di Sistema (Hotkey)](#7-passo-4-scorciatoia-globale-di-sistema-hotkey)
8. [Passo 5: Comandi Avanzati da Riga di Comando (CLI)](#8-passo-5-comandi-avanzati-da-riga-di-comando-cli)
9. [Troubleshooting & FAQ](#9-troubleshooting--faq)

---

## 1. Panoramica dell'Architettura

GemFinder unisce:
- **Inference Engine (`llama.cpp` nativo):** esegue EmbeddingGemma 2 con accelerazione GPU (Vulkan/ROCm/CUDA o CPU AVX2) estraendo embedding con *Matryoshka Representation Learning (MRL)* troncati a **256 dimensioni** con normalizzazione $L_2$.
- **Storage Layer Ibrido:** SQLite3 in modalità WAL con estensione **FTS5 (BM25)** per le ricerche lessicali e **USearch (HNSW)** per la ricerca vettoriale ad altissima velocità.
- **Fusione RRF (Reciprocal Rank Fusion):** combina le rilevanze lessicali e semantiche secondo la formula:
  $$\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{vettore}}} + \frac{1}{60 + \text{rank}_{\text{bm25}}}$$
- **GUI Desktop Raycast/Spotlight (Qt 6):** interfaccia a comparsa rapida che **raggruppa automaticamente i risultati per Testo, Immagini e Audio**.

```
  ┌──────────────────────────────────────────────────────────────┐
  │                 gemfinder-gui (Qt 6)                         │
  │    [📄 Testo & Codice]   [🖼️ Immagini]   [🎵 Audio]           │
  └──────────────────────────────┬───────────────────────────────┘
                                 │ Unix Domain Socket (/tmp/gemfinder.sock)
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                 gemfinder-daemon (C++20)                     │
  │  ┌──────────────────────┐        ┌────────────────────────┐  │
  │  │ inotify FileWatcher  │        │   EmbeddingGemma 2     │  │
  │  │ & Concurrency Queue  │        │    (llama.cpp GGUF)    │  │
  │  └──────────┬───────────┘        └───────────┬────────────┘  │
  │             ▼                                ▼               │
  │  ┌────────────────────────────────────────────────────────┐  │
  │  │   StorageManager: SQLite3 WAL (FTS5) + USearch (HNSW)   │  │
  │  └────────────────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────────────────┘
```

---

## 2. Requisiti e Download di EmbeddingGemma 2 GGUF

GemFinder carica nativamente qualsiasi modello di embedding in formato **GGUF** compatibile con `llama.cpp`.

### Dove scaricare i pesi GGUF
Crea una directory per i modelli:
```bash
mkdir -p ~/.local/share/gemfinder/models
cd ~/.local/share/gemfinder/models
```

Puoi scaricare il modello **EmbeddingGemma 2** (o altri modelli compatibili come `bge-m3` o `nomic-embed-text`) direttamente da Hugging Face usando `curl`, `wget` o `huggingface-cli`:

```bash
# Esempio di download con curl (sostituire con l'URL o repo GGUF desiderato):
curl -L -o embeddinggemma-2-2b-q8_0.gguf \
  "https://huggingface.co/google/embeddinggemma-2b-GGUF/resolve/main/embeddinggemma-2-2b-q8_0.gguf"
```

> **Nota di Fallback:** Se non specifichi il flag `--model`, GemFinder si avvia istantaneamente utilizzando il **`DeterministicMockEngine`** integrato. Questo consente di indicizzare e testare tutte le funzionalità (inclusi FTS5, ranking RRF, daemon e GUI) senza dover scaricare alcun modello.

---

## 3. Compilazione e Installazione

Se non hai ancora compilato il binario in modalità Release:

```bash
cd /home/pietro/Documents/Code/GemFinder
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Per installare i binari e i file di sistema nel percorso standard (`/usr/local/bin`):
```bash
sudo cmake --install build
```

Verranno installati:
- `gemfinder-cli` (Strumento da terminale)
- `gemfinder-daemon` (Servizio di background)
- `gemfinder-gui` (Interfaccia grafica Desktop)
- `gemfinder-benchmark` (Suite di test delle prestazioni)
- `gemfinder-toggle` (Script di attivazione rapida per scorciatoie)

---

## 4. Passo 1: Indicizzazione Iniziale dei Documenti (CLI)

Prima di avviare le ricerche, esegui una prima scansione delle cartelle che desideri indicizzare (es. `~/Documents`, `~/Pictures`, `~/Musica` o codice sorgente):

```bash
./build/gemfinder-cli index ~/Documents ~/Pictures \
  --model ~/.local/share/gemfinder/models/embeddinggemma-2-2b-q8_0.gguf
```

### Cosa fa GemFinder durante la scansione:
1. **Esclusione automatica:** salta cartelle di build, `.git`, `node_modules`, cache e file temporanei.
2. **Preprocessing Multimodale:**
   - **Testo (`.txt`, `.md`, `.cpp`, `.py`, ecc.):** frammenta in paragrafi con sliding-window mantenendo i numeri di riga esatti (`start_line`/`end_line`).
   - **Immagini (`.png`, `.jpg`, `.webp`, ecc.):** decodifica con `stb_image` e ridimensionamento bilineare a 224x224.
   - **Audio (`.wav`, `.mp3`, `.flac`, ecc.):** ricampionamento a 16 kHz Mono float32 con finestre da 30 secondi e 5 secondi di sovrapposizione.
3. **Scansione Incrementale Intelligente:** calcola `mtime` e `xxHash64`. Nelle esecuzioni successive, i file non modificati vengono saltati a costo zero.

---

## 5. Passo 2: Avvio del Demone di Background (`gemfinder-daemon`)

Il demone mantiene il modello EmbeddingGemma 2 e l'indice USearch caldi in memoria RAM/VRAM e riceve eventi live del filesystem tramite `inotify`.

### Avvio Manuale da Terminale
```bash
./build/gemfinder-daemon \
  --model ~/.local/share/gemfinder/models/embeddinggemma-2-2b-q8_0.gguf \
  --watch ~/Documents \
  --watch ~/Pictures
```

### Avvio Automatico con Systemd (Consigliato)
1. Copia l'unità utente:
   ```bash
   mkdir -p ~/.config/systemd/user/
   cp systemd/gemfinder-daemon.service ~/.config/systemd/user/
   ```
2. Modifica il percorso del modello nel file `~/.config/systemd/user/gemfinder-daemon.service` aggiungendo `--model ~/.local/share/gemfinder/models/embeddinggemma-2-2b-q8_0.gguf`.
3. Abilita e avvia il servizio:
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now gemfinder-daemon
   ```
4. Verifica lo stato:
   ```bash
   systemctl --user status gemfinder-daemon
   ```

---

## 6. Passo 3: Ricerca Desktop con GUI Spotlight (`gemfinder-gui`)

Avvia l'interfaccia utente grafica:
```bash
./build/gemfinder-gui
```

### Funzionalità dell'Interfaccia:
- **Barra di Ricerca in Tempo Reale:** digita la tua query concettuale (es. *"contratto di locazione appartamento"*, *"paesaggio di montagna innevata"*, *"assolo di chitarra rock"*). La ricerca parte automaticamente dopo 150ms di inattività dalla digitazione.
- **Raggruppamento Rigoroso per Modalità:**
  I risultati compaiono divisi in 3 eleganti sezioni visuali:
  1. 📄 **DOCUMENTI & TESTO:** mostra il nome del file, il percorso, la riga d'inizio (`Riga L42`), l'anteprima formattata del testo e la percentuale di match RRF.
  2. 🖼️ **IMMAGINI:** visualizza direttamente l'anteprima (*thumbnail*) dell'immagine scalata, le dimensioni e la somiglianza.
  3. 🎵 **AUDIO & REGISTRAZIONI:** mostra l'icona audio, il minutaggio temporale del frammento (`⏱ 01:30 - 02:00`) e la rilevanza.
- **Schede di Navigazione Rapida:**
  Puoi cliccare sui pulsanti a pillola in alto per filtrare:
  - `🔘 Tutti (Raggruppati)`: visualizzazione completa suddivisa per categorie.
  - `📄 Testo (N)`: visualizza solo i risultati testuali.
  - `🖼️ Immagini (N)`: visualizza solo le immagini trovate.
  - `🎵 Audio (N)`: visualizza solo i file audio.
- **Ricerca Multimodale via Drag & Drop:**
  - Trascina un'immagine dalla tua scrivania o dal file manager direttamente nella finestra di GemFinder: partirà immediatamente la ricerca per similarità visiva.
  - Trascina un file `.mp3` o `.wav`: cercherà istantaneamente spezzoni sonori analoghi.
- **Interazione Diretta:**
  - **Doppio clic o Invio:** apre il file selezionato con l'applicazione di sistema predefinita.
  - **Tasto destro:** menu contestuale con opzioni *"Apri File"*, *"Mostra nella Cartella"*, *"Copia Percorso"* e *"Copia Testo Anteprima"*.
  - **Tasto Escape:** chiude istantaneamente la finestra.

---

## 7. Passo 4: Scorciatoia Globale di Sistema (Hotkey)

Per richiamare GemFinder in qualsiasi momento (in stile Spotlight su macOS o Raycast) associandolo ad esempio a **`Super + Spazio`** o **`Alt + Spazio`**:

### GNOME Desktop
1. Apri **Impostazioni** -> **Tastiera** -> **Visualizza e personalizza scorciatoie** -> **Scorciatoie personalizzate**.
2. Fai clic su **+** (Aggiungi scorciatoia):
   - **Nome:** `GemFinder`
   - **Comando:** `/usr/local/bin/gemfinder-toggle` (oppure `/home/pietro/Documents/Code/GemFinder/scripts/gemfinder-toggle.sh`)
   - **Scorciatoia:** `Super + Spazio` o `Ctrl + Spazio`

### KDE Plasma
1. Apri **Impostazioni di Sistema** -> **Scorciatoie** -> **Comandi personalizzati**.
2. Aggiungi nuovo comando, imposta come comando `gemfinder-toggle` e assegna la combinazione desiderata.

### Hyprland (`~/.config/hypr/hyprland.conf`)
Aggiungi la seguente riga:
```ini
bind = $mainMod, SPACE, exec, /usr/local/bin/gemfinder-toggle
```

### Sway (`~/.config/sway/config`)
Aggiungi:
```ini
bindsym $mod+space exec /usr/local/bin/gemfinder-toggle
```

---

## 8. Passo 5: Comandi Avanzati da Riga di Comando (CLI)

Se preferisci lavorare da terminale o integrare GemFinder in script bash:

### Ricerca Testuale Ibrida
```bash
./build/gemfinder-cli search "bilancio annuale e fatture" --limit 10
```

### Ricerca per Immagine Query
```bash
./build/gemfinder-cli search --image ~/Downloads/foto_riferimento.jpg --limit 5
```

### Ricerca per Audio Query
```bash
./build/gemfinder-cli search --audio ~/Recordings/nota_vocale.wav --limit 5
```

### Statistiche dell'Indice
```bash
./build/gemfinder-cli stats
```
Output tipico:
```
=== GemFinder Index Statistics ===
File indicizzati:    4,189
Chunk totali:        115,841
Vettori in USearch:  115,841
Dimensione Vettori:  256 floats
Modello Inferenza:   embeddinggemma-2-2b-q8_0.gguf
DB Metadati:         "gemfinder_metadata.db"
Indice Vettoriale:   "gemfinder_metadata.usearch"
```

### Ispezione di un File Singolo
Per vedere come GemFinder decodifica e suddivide in chunk un file:
```bash
./build/gemfinder-cli inspect documento.pdf
```

### Calcolo Vettore Diretto
```bash
./build/gemfinder-cli embed "testo di prova per embedding"
```

---

## 9. Troubleshooting & FAQ

### D: Cosa succede se non uso il flag `--model`?
**R:** GemFinder utilizza automaticamente il motore deterministico integrato `DeterministicMockEngine`. Tutte le feature (USearch HNSW, SQLite FTS5, ranking RRF, daemon e GUI) funzionano regolarmente a latenze sub-millisecondo. Quando colleghi il modello GGUF reale di EmbeddingGemma 2, la qualità semantica e concettuale dei match diventa quella dello stato dell'arte dei modelli linguistici di Google.

### D: Dove vengono salvati gli indici e il database?
**R:** Di default nella cartella corrente con i nomi `gemfinder_metadata.db` (SQLite in modalità WAL) e `gemfinder_metadata.usearch` (HNSW). Puoi indicare un percorso personalizzato usando il parametro `--db /percorso/a/database.db`.

### D: La GUI funziona se il demone è spento?
**R:** **Sì.** `gemfinder-gui` controlla prima la presenza del socket `/tmp/gemfinder.sock`. Se il demone è attivo, si collega via IPC a latenza zero (< 0.5 ms); se il demone è spento, la GUI apre in locale i file dell'indice senza interrompere l'utente (mostrando il badge `🟠 Indice Locale`).

### D: Come posso verificare le prestazioni del mio hardware?
**R:** Esegui il tool di benchmark dedicato:
```bash
./build/gemfinder-benchmark 20000
```
Misurerà il throughput di indicizzazione (chunk/sec), le latenze percentili (p50, p95, p99), il consumo effettivo di memoria RAM e il throughput sotto carico concorrente multi-thread.
