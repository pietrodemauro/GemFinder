#include "gui/MainWindow.hpp"
#include "gui/SearchResultItem.hpp"

#include <QApplication>
#include <QScreen>
#include <QFileDialog>
#include <QMimeData>
#include <QKeyEvent>
#include <QMessageBox>
#include <iomanip>
#include <sstream>

namespace gemfinder {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      search_worker_(new SearchWorker(this)),
      debounce_timer_(new QTimer(this)) {
    setupUi();
    setupStyles();

    // Center on active screen
    resize(980, 720);
    if (const QScreen* scr = window()->screen()) {
        const QRect geom = scr->availableGeometry();
        move((geom.width() - width()) / 2, (geom.height() - height()) / 3);
    }

    setAcceptDrops(true);

    debounce_timer_->setSingleShot(true);
    debounce_timer_->setInterval(150); // 150ms debounce

    connect(search_input_, &QLineEdit::textChanged, this, &MainWindow::onSearchTextChanged);
    connect(debounce_timer_, &QTimer::timeout, this, &MainWindow::onDebounceTimeout);
    connect(search_worker_, &SearchWorker::searchCompleted, this, &MainWindow::onSearchCompleted);
    connect(search_worker_, &SearchWorker::searchError, this, &MainWindow::onSearchError);
    connect(search_worker_, &SearchWorker::daemonStatusChanged, this, &MainWindow::onDaemonStatusChanged);

    connect(btn_image_search_, &QPushButton::clicked, this, &MainWindow::onSelectImageQuery);
    connect(btn_audio_search_, &QPushButton::clicked, this, &MainWindow::onSelectAudioQuery);

    connect(btn_tab_all_, &QPushButton::clicked, this, [this]() { onFilterChanged(DisplayFilter::AllGrouped); });
    connect(btn_tab_text_, &QPushButton::clicked, this, [this]() { onFilterChanged(DisplayFilter::TextOnly); });
    connect(btn_tab_image_, &QPushButton::clicked, this, [this]() { onFilterChanged(DisplayFilter::ImageOnly); });
    connect(btn_tab_audio_, &QPushButton::clicked, this, [this]() { onFilterChanged(DisplayFilter::AudioOnly); });

    // Initial check
    onDaemonStatusChanged(search_worker_->isDaemonConnected());

    // Initial query empty state
    renderResults();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUi() {
    setWindowTitle("GemFinder — Ricerca Semantica Multimodale");

    auto* central = new QWidget(this);
    setCentralWidget(central);

    auto* main_layout = new QVBoxLayout(central);
    main_layout->setContentsMargins(20, 20, 20, 16);
    main_layout->setSpacing(12);

    // --- TOP BAR: Search Input + Multimodal Buttons + Daemon Badge ---
    auto* top_row = new QHBoxLayout();
    top_row->setSpacing(10);

    search_input_ = new QLineEdit(this);
    search_input_->setPlaceholderText("🔍 Cerca per significato (es. \"contratto affitto\", \"paesaggio tramonto\", \"assolo rock\")...");
    search_input_->setClearButtonEnabled(true);
    top_row->addWidget(search_input_, 1);

    btn_image_search_ = new QPushButton("🖼️ Immagine...", this);
    btn_image_search_->setToolTip("Cerca file simili usando un'immagine query");
    top_row->addWidget(btn_image_search_);

    btn_audio_search_ = new QPushButton("🎵 Audio...", this);
    btn_audio_search_->setToolTip("Cerca file simili usando un file audio query");
    top_row->addWidget(btn_audio_search_);

    daemon_status_badge_ = new QLabel("⚪ Controllo...", this);
    top_row->addWidget(daemon_status_badge_);

    main_layout->addLayout(top_row);

    // --- FILTER TABS BAR ---
    auto* filter_bar = new QHBoxLayout();
    filter_bar->setSpacing(8);

    btn_tab_all_ = new QPushButton("🔘 Tutti (Raggruppati)", this);
    btn_tab_all_->setCheckable(true);
    btn_tab_all_->setChecked(true);
    filter_bar->addWidget(btn_tab_all_);

    btn_tab_text_ = new QPushButton("📄 Testo (0)", this);
    btn_tab_text_->setCheckable(true);
    filter_bar->addWidget(btn_tab_text_);

    btn_tab_image_ = new QPushButton("🖼️ Immagini (0)", this);
    btn_tab_image_->setCheckable(true);
    filter_bar->addWidget(btn_tab_image_);

    btn_tab_audio_ = new QPushButton("🎵 Audio (0)", this);
    btn_tab_audio_->setCheckable(true);
    filter_bar->addWidget(btn_tab_audio_);

    filter_bar->addStretch();
    main_layout->addLayout(filter_bar);

    // --- SCROLLABLE RESULTS CONTAINER ---
    scroll_area_ = new QScrollArea(this);
    scroll_area_->setWidgetResizable(true);
    scroll_area_->setFrameShape(QFrame::NoFrame);

    results_container_ = new QWidget();
    results_layout_ = new QVBoxLayout(results_container_);
    results_layout_->setContentsMargins(4, 4, 4, 4);
    results_layout_->setSpacing(6);
    results_layout_->addStretch();

    scroll_area_->setWidget(results_container_);
    main_layout->addWidget(scroll_area_, 1);

    // --- BOTTOM STATS BAR ---
    stats_bar_ = new QLabel("Pronto per la ricerca semantica. Trascina un'immagine o un audio per ricerca multimodale.", this);
    main_layout->addWidget(stats_bar_);
}

void MainWindow::setupStyles() {
    setStyleSheet(R"(
        QMainWindow {
            background-color: #181926;
        }
        QLineEdit {
            background-color: #24273a;
            color: #cad3f5;
            border: 2px solid #363a4f;
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 14px;
            selection-background-color: #89b4fa;
            selection-color: #181926;
        }
        QLineEdit:focus {
            border: 2px solid #89b4fa;
            background-color: #1e2030;
        }
        QPushButton {
            background-color: #24273a;
            color: #cad3f5;
            border: 1px solid #363a4f;
            border-radius: 8px;
            padding: 8px 14px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #363a4f;
            border: 1px solid #89b4fa;
            color: #89b4fa;
        }
        QPushButton:checked {
            background-color: #89b4fa;
            color: #181926;
            font-weight: bold;
            border: 1px solid #89b4fa;
        }
        QScrollArea {
            background: transparent;
            border: none;
        }
        QScrollBar:vertical {
            background: #181926;
            width: 10px;
            margin: 0px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: #363a4f;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: #89b4fa;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QLabel {
            color: #a5adcb;
            font-size: 12px;
        }
    )");

    daemon_status_badge_->setStyleSheet("font-size: 11px; padding: 4px 10px; border-radius: 8px; background-color: #24273a; font-weight: bold;");
}

void MainWindow::onSearchTextChanged(const QString& /*text*/) {
    debounce_timer_->start();
}

void MainWindow::onDebounceTimeout() {
    triggerSearch();
}

void MainWindow::triggerSearch() {
    QString q = search_input_->text().trimmed();
    if (q.isEmpty()) {
        all_results_.clear();
        text_results_.clear();
        image_results_.clear();
        audio_results_.clear();
        renderResults();
        stats_bar_->setText("Pronto per la ricerca.");
        return;
    }

    IpcRequest req;
    req.action = IpcAction::SearchText;
    req.query = q.toStdString();
    req.limit = 40; // Retrieve enough results to populate all modalities

    active_query_mode_ = "Testo (" + q + ")";
    stats_bar_->setText(QString("Ricerca in corso per \"%1\"...").arg(q));
    search_worker_->requestSearch(req);
}

void MainWindow::onSelectImageQuery() {
    QString path = QFileDialog::getOpenFileName(
        this,
        "Seleziona un'immagine per ricerca per similarità",
        "",
        "Immagini (*.png *.jpg *.jpeg *.bmp *.webp *.gif)"
    );

    if (path.isEmpty()) return;

    search_input_->setText(QString("[Immagine: %1]").arg(QFileInfo(path).fileName()));
    active_query_mode_ = "Immagine (" + QFileInfo(path).fileName() + ")";

    IpcRequest req;
    req.action = IpcAction::SearchImage;
    req.path = path.toStdString();
    req.limit = 40;

    stats_bar_->setText(QString("Ricerca per similarità visiva con \"%1\"...").arg(path));
    search_worker_->requestSearch(req);
}

void MainWindow::onSelectAudioQuery() {
    QString path = QFileDialog::getOpenFileName(
        this,
        "Seleziona un file audio per ricerca semantica",
        "",
        "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac)"
    );

    if (path.isEmpty()) return;

    search_input_->setText(QString("[Audio: %1]").arg(QFileInfo(path).fileName()));
    active_query_mode_ = "Audio (" + QFileInfo(path).fileName() + ")";

    IpcRequest req;
    req.action = IpcAction::SearchAudio;
    req.path = path.toStdString();
    req.limit = 40;

    stats_bar_->setText(QString("Ricerca per similarità acustica con \"%1\"...").arg(path));
    search_worker_->requestSearch(req);
}

void MainWindow::onSearchCompleted(const IpcResponse& response) {
    all_results_ = response.results;
    last_latency_ms_ = response.latency_ms;

    // Partition results by modality
    text_results_.clear();
    image_results_.clear();
    audio_results_.clear();

    for (const auto& item : all_results_) {
        Modality m = detectResultModality(item);
        if (m == Modality::Text) {
            text_results_.push_back(item);
        } else if (m == Modality::Image) {
            image_results_.push_back(item);
        } else if (m == Modality::Audio) {
            audio_results_.push_back(item);
        }
    }

    // Update filter tab labels
    btn_tab_text_->setText(QString("📄 Testo (%1)").arg(text_results_.size()));
    btn_tab_image_->setText(QString("🖼️ Immagini (%1)").arg(image_results_.size()));
    btn_tab_audio_->setText(QString("🎵 Audio (%1)").arg(audio_results_.size()));

    renderResults();

    stats_bar_->setText(QString("⚡ %1 risultati trovati in %2 ms | %3")
        .arg(all_results_.size())
        .arg(last_latency_ms_, 0, 'f', 2)
        .arg(active_query_mode_));
}

void MainWindow::onSearchError(const QString& error_msg) {
    stats_bar_->setText(QString("⚠️ Errore di ricerca: %1").arg(error_msg));
}

void MainWindow::onDaemonStatusChanged(bool is_running) {
    if (is_running) {
        daemon_status_badge_->setText("🟢 Demone Connesso");
        daemon_status_badge_->setStyleSheet("font-size: 11px; padding: 4px 10px; border-radius: 8px; background-color: #1e3a29; color: #a6da95; font-weight: bold;");
    } else {
        daemon_status_badge_->setText("🟠 Indice Locale");
        daemon_status_badge_->setStyleSheet("font-size: 11px; padding: 4px 10px; border-radius: 8px; background-color: #3d2b1f; color: #f5a97f; font-weight: bold;");
    }
}

void MainWindow::onFilterChanged(DisplayFilter filter) {
    current_filter_ = filter;

    btn_tab_all_->setChecked(filter == DisplayFilter::AllGrouped);
    btn_tab_text_->setChecked(filter == DisplayFilter::TextOnly);
    btn_tab_image_->setChecked(filter == DisplayFilter::ImageOnly);
    btn_tab_audio_->setChecked(filter == DisplayFilter::AudioOnly);

    renderResults();
}

void MainWindow::clearResultsLayout() {
    // Delete all widgets in results_layout_
    QLayoutItem* item = nullptr;
    while ((item = results_layout_->takeAt(0)) != nullptr) {
        if (item->widget()) {
            delete item->widget();
        }
        delete item;
    }
}

void MainWindow::renderResults() {
    clearResultsLayout();

    if (all_results_.empty()) {
        auto* empty_label = new QLabel(
            "Nessun risultato da mostrare.\n"
            "Inizia a digitare per cercare, oppure trascina un file di immagine o audio nella finestra.",
            results_container_);
        empty_label->setAlignment(Qt::AlignCenter);
        empty_label->setStyleSheet("color: #6e738d; font-size: 14px; margin-top: 60px; line-height: 1.5;");
        results_layout_->addWidget(empty_label);
        results_layout_->addStretch();
        return;
    }

    auto renderGroup = [this](Modality modality, const std::vector<SearchResult>& items) {
        if (items.empty()) return;

        auto* header = new CategoryHeader(modality, items.size(), results_container_);
        results_layout_->addWidget(header);

        for (const auto& item : items) {
            auto* card = new ResultCard(item, results_container_);
            results_layout_->addWidget(card);
        }
    };

    if (current_filter_ == DisplayFilter::AllGrouped) {
        // Grouped by Text, Images, and Audio as requested!
        renderGroup(Modality::Text, text_results_);
        renderGroup(Modality::Image, image_results_);
        renderGroup(Modality::Audio, audio_results_);
    } else if (current_filter_ == DisplayFilter::TextOnly) {
        renderGroup(Modality::Text, text_results_);
    } else if (current_filter_ == DisplayFilter::ImageOnly) {
        renderGroup(Modality::Image, image_results_);
    } else if (current_filter_ == DisplayFilter::AudioOnly) {
        renderGroup(Modality::Audio, audio_results_);
    }

    results_layout_->addStretch();
}

void MainWindow::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Escape) {
        close();
    } else if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
        if (search_input_->hasFocus()) {
            debounce_timer_->stop();
            triggerSearch();
        }
    }
    QMainWindow::keyPressEvent(event);
}

void MainWindow::dragEnterEvent(QDragEnterEvent* event) {
    if (event->mimeData()->hasUrls()) {
        event->acceptProposedAction();
    }
}

void MainWindow::dropEvent(QDropEvent* event) {
    const auto urls = event->mimeData()->urls();
    if (urls.isEmpty()) return;

    QString file_path = urls.first().toLocalFile();
    if (file_path.isEmpty()) return;

    QFileInfo fi(file_path);
    QString ext = fi.suffix().toLower();

    if (ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "bmp" || ext == "webp") {
        search_input_->setText(QString("[Immagine: %1]").arg(fi.fileName()));
        active_query_mode_ = "Immagine (" + fi.fileName() + ")";
        IpcRequest req;
        req.action = IpcAction::SearchImage;
        req.path = file_path.toStdString();
        req.limit = 40;
        stats_bar_->setText(QString("Ricerca per immagine trascinata: %1...").arg(fi.fileName()));
        search_worker_->requestSearch(req);
    } else if (ext == "wav" || ext == "mp3" || ext == "flac" || ext == "ogg" || ext == "m4a") {
        search_input_->setText(QString("[Audio: %1]").arg(fi.fileName()));
        active_query_mode_ = "Audio (" + fi.fileName() + ")";
        IpcRequest req;
        req.action = IpcAction::SearchAudio;
        req.path = file_path.toStdString();
        req.limit = 40;
        stats_bar_->setText(QString("Ricerca per audio trascinato: %1...").arg(fi.fileName()));
        search_worker_->requestSearch(req);
    } else {
        // Assume text file
        search_input_->setText(fi.baseName());
        triggerSearch();
    }
}

} // namespace gemfinder
