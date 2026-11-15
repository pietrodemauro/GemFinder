#include "gui/MainWindow.hpp"
#include "gui/SearchResultItem.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QFileInfo>
#include <QDir>
#include <QScrollBar>
#include <QDesktopServices>
#include <QProcess>
#include <algorithm>

namespace gemfinder {

MainWindow::MainWindow(std::optional<fs::path> model_path, int n_threads, QWidget* parent)
    : QMainWindow(parent),
      search_worker_(new SearchWorker(std::move(model_path), n_threads, true, this)),
      debounce_timer_(new QTimer(this)) {

    setupUi();
    setupStyles();

    setAcceptDrops(true);
    resize(860, 640);
    setMinimumSize(700, 500);

    // Setup Debounce Timer (200 ms)
    debounce_timer_->setSingleShot(true);
    debounce_timer_->setInterval(200);
    connect(debounce_timer_, &QTimer::timeout, this, &MainWindow::onDebounceTimeout);

    // SearchWorker Signals
    connect(search_worker_, &SearchWorker::searchCompleted, this, &MainWindow::onSearchCompleted);
    connect(search_worker_, &SearchWorker::searchError, this, &MainWindow::onSearchError);
    connect(search_worker_, &SearchWorker::daemonStatusChanged, this, &MainWindow::onDaemonStatusChanged);

    // Search Input Signals
    connect(search_input_, &QLineEdit::textChanged, this, &MainWindow::onSearchTextChanged);

    // Button Signals
    connect(btn_image_search_, &QPushButton::clicked, this, &MainWindow::onSelectImageQuery);
    connect(btn_audio_search_, &QPushButton::clicked, this, &MainWindow::onSelectAudioQuery);
    connect(btn_index_dir_, &QPushButton::clicked, this, &MainWindow::onIndexDirectory);
    connect(btn_unindex_dir_, &QPushButton::clicked, this, &MainWindow::onUnindexDirectory);

    // Filter Buttons Signals
    connect(btn_tab_all_, &QPushButton::clicked, this, [this]() { onFilterChanged(DisplayFilter::AllGrouped); });
    connect(btn_tab_text_, &QPushButton::clicked, this, [this]() { onFilterChanged(DisplayFilter::TextOnly); });
    connect(btn_tab_image_, &QPushButton::clicked, this, [this]() { onFilterChanged(DisplayFilter::ImageOnly); });
    connect(btn_tab_audio_, &QPushButton::clicked, this, [this]() { onFilterChanged(DisplayFilter::AudioOnly); });

    // Hardware & Model Control Signals
    connect(combo_device_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onDeviceChanged);
    connect(btn_change_model_, &QPushButton::clicked, this, &MainWindow::onChangeModel);
    connect(slider_threshold_, &QSlider::valueChanged, this, &MainWindow::onThresholdChanged);

    label_model_name_->setText(QString::fromStdString(search_worker_->currentModelName()));

    // Initial check
    onDaemonStatusChanged(search_worker_->isDaemonConnected());

    // Initial query empty state
    renderResults();
}

MainWindow::~MainWindow() = default;

void MainWindow::setupUi() {
    setWindowTitle("GemFinder — Multimodal Semantic Search");

    auto* central = new QWidget(this);
    setCentralWidget(central);

    auto* main_layout = new QVBoxLayout(central);
    main_layout->setContentsMargins(20, 20, 20, 16);
    main_layout->setSpacing(12);

    // --- TOP BAR: Search Input + Multimodal Buttons + Daemon Badge ---
    auto* top_row = new QHBoxLayout();
    top_row->setSpacing(10);

    search_input_ = new QLineEdit(this);
    search_input_->setPlaceholderText("🔍 Search by meaning (e.g. \"lease contract\", \"sunset landscape\", \"guitar solo\")...");
    search_input_->setClearButtonEnabled(true);
    top_row->addWidget(search_input_, 1);

    btn_image_search_ = new QPushButton("🖼️ Image...", this);
    btn_image_search_->setToolTip("Search similar files using a query image");
    top_row->addWidget(btn_image_search_);

    btn_audio_search_ = new QPushButton("🎵 Audio...", this);
    btn_audio_search_->setToolTip("Search similar files using a query audio file");
    top_row->addWidget(btn_audio_search_);

    daemon_status_badge_ = new QLabel("⚪ Checking...", this);
    top_row->addWidget(daemon_status_badge_);

    main_layout->addLayout(top_row);

    // --- FILTER TABS BAR ---
    auto* filter_bar = new QHBoxLayout();
    filter_bar->setSpacing(8);

    btn_tab_all_ = new QPushButton("🔘 All (Grouped)", this);
    btn_tab_all_->setCheckable(true);
    btn_tab_all_->setChecked(true);
    filter_bar->addWidget(btn_tab_all_);

    btn_tab_text_ = new QPushButton("📄 Text (0)", this);
    btn_tab_text_->setCheckable(true);
    filter_bar->addWidget(btn_tab_text_);

    btn_tab_image_ = new QPushButton("🖼️ Images (0)", this);
    btn_tab_image_->setCheckable(true);
    filter_bar->addWidget(btn_tab_image_);

    btn_tab_audio_ = new QPushButton("🎵 Audio (0)", this);
    btn_tab_audio_->setCheckable(true);
    filter_bar->addWidget(btn_tab_audio_);

    filter_bar->addStretch();

    btn_index_dir_ = new QPushButton("📁 Index Directory...", this);
    btn_index_dir_->setToolTip("Select and index a directory into local database / daemon");
    filter_bar->addWidget(btn_index_dir_);

    btn_unindex_dir_ = new QPushButton("🗑️ Remove Directory...", this);
    btn_unindex_dir_->setToolTip("Remove indexed directory from database to re-index from scratch");
    filter_bar->addWidget(btn_unindex_dir_);

    main_layout->addLayout(filter_bar);

    // --- HARDWARE & MODEL SETTINGS BAR ---
    auto* settings_bar = new QHBoxLayout();
    settings_bar->setSpacing(8);

    auto* lbl_device = new QLabel("⚡ Device:", this);
    lbl_device->setStyleSheet("color: #a5adcb; font-size: 12px; font-weight: bold;");
    settings_bar->addWidget(lbl_device);

    combo_device_ = new QComboBox(this);
    combo_device_->addItem("🚀 GPU (Vulkan)", true);
    combo_device_->addItem("💻 CPU (Multi-core)", false);
    combo_device_->setToolTip("Select compute device for local embedding inference");
    settings_bar->addWidget(combo_device_);

    settings_bar->addSpacing(12);

    auto* lbl_model = new QLabel("🧠 Model:", this);
    lbl_model->setStyleSheet("color: #a5adcb; font-size: 12px; font-weight: bold;");
    settings_bar->addWidget(lbl_model);

    label_model_name_ = new QLabel("embeddinggemma-2.gguf", this);
    label_model_name_->setStyleSheet("background-color: #24273a; color: #89b4fa; border: 1px solid #363a4f; border-radius: 6px; padding: 4px 8px; font-size: 12px; font-weight: bold;");
    label_model_name_->setToolTip("Current GGUF embedding model");
    settings_bar->addWidget(label_model_name_);

    btn_change_model_ = new QPushButton("Browse Model (.gguf)...", this);
    btn_change_model_->setToolTip("Select a local GGUF embedding model file");
    settings_bar->addWidget(btn_change_model_);

    settings_bar->addSpacing(16);

    auto* lbl_threshold = new QLabel("🎯 Min Similarity:", this);
    lbl_threshold->setStyleSheet("color: #a5adcb; font-size: 12px; font-weight: bold;");
    settings_bar->addWidget(lbl_threshold);

    slider_threshold_ = new QSlider(Qt::Horizontal, this);
    slider_threshold_->setRange(0, 95);
    slider_threshold_->setValue(0);
    slider_threshold_->setFixedWidth(120);
    slider_threshold_->setToolTip("Minimum similarity threshold to filter results (0% to 95%)");
    settings_bar->addWidget(slider_threshold_);

    label_threshold_val_ = new QLabel("0%", this);
    label_threshold_val_->setStyleSheet("background-color: #24273a; color: #a6da95; border: 1px solid #363a4f; border-radius: 6px; padding: 4px 8px; font-size: 12px; font-weight: bold;");
    label_threshold_val_->setFixedWidth(45);
    label_threshold_val_->setAlignment(Qt::AlignCenter);
    settings_bar->addWidget(label_threshold_val_);

    settings_bar->addStretch();
    main_layout->addLayout(settings_bar);

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
    stats_bar_ = new QLabel("Ready for semantic search. Drag and drop an image or audio file for multimodal search.", this);
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
        QComboBox {
            background-color: #24273a;
            color: #cad3f5;
            border: 1px solid #363a4f;
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 500;
            min-width: 140px;
        }
        QComboBox:hover {
            border: 1px solid #89b4fa;
        }
        QComboBox::drop-down {
            border: none;
            padding-right: 8px;
        }
        QComboBox QAbstractItemView {
            background-color: #1e2030;
            color: #cad3f5;
            selection-background-color: #363a4f;
            selection-color: #89b4fa;
            border: 1px solid #363a4f;
            border-radius: 6px;
            padding: 4px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #363a4f;
            height: 6px;
            background: #24273a;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: #89b4fa;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #cad3f5;
            border: 1px solid #89b4fa;
            width: 14px;
            margin-top: -5px;
            margin-bottom: -5px;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: #89b4fa;
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
        raw_results_.clear();
        all_results_.clear();
        text_results_.clear();
        image_results_.clear();
        audio_results_.clear();
        renderResults();
        stats_bar_->setText("Ready for search.");
        return;
    }

    IpcRequest req;
    req.action = IpcAction::SearchText;
    req.query = q.toStdString();
    req.limit = 40; // Retrieve enough results to populate all modalities
    req.threshold = current_threshold_;

    active_query_mode_ = "Text (" + q + ")";
    stats_bar_->setText(QString("Searching for \"%1\"...").arg(q));
    search_worker_->requestSearch(req);
}

void MainWindow::onSelectImageQuery() {
    QString path = QFileDialog::getOpenFileName(
        this,
        "Select an image for similarity search",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.webp *.gif)"
    );

    if (path.isEmpty()) return;

    search_input_->setText(QString("[Image: %1]").arg(QFileInfo(path).fileName()));
    active_query_mode_ = "Image (" + QFileInfo(path).fileName() + ")";

    IpcRequest req;
    req.action = IpcAction::SearchImage;
    req.path = path.toStdString();
    req.limit = 40;
    req.threshold = current_threshold_;

    stats_bar_->setText(QString("Visual similarity search for \"%1\"...").arg(path));
    search_worker_->requestSearch(req);
}

void MainWindow::onSelectAudioQuery() {
    QString path = QFileDialog::getOpenFileName(
        this,
        "Select an audio file for semantic search",
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
    req.threshold = current_threshold_;

    stats_bar_->setText(QString("Acoustic similarity search for \"%1\"...").arg(path));
    search_worker_->requestSearch(req);
}

void MainWindow::onIndexDirectory() {
    QString dir = QFileDialog::getExistingDirectory(
        this,
        "Select directory to index",
        QDir::homePath(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );

    if (dir.isEmpty()) return;

    stats_bar_->setText(QString("⏳ Indexing in progress for \"%1\"...").arg(dir));
    IpcRequest req;
    req.action = IpcAction::IndexDirectory;
    req.path = dir.toStdString();
    search_worker_->requestAction(req);
}

void MainWindow::onUnindexDirectory() {
    QString dir = QFileDialog::getExistingDirectory(
        this,
        "Select directory to remove from index",
        QDir::homePath(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );

    if (dir.isEmpty()) return;

    auto reply = QMessageBox::question(
        this,
        "Confirm Index Removal",
        QString("Are you sure you want to remove the index for directory:\n\n%1\n\nIndexed files and associated vectors will be deleted from the database to allow clean re-indexing.\n(Your original files on disk will NOT be touched).").arg(dir),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No
    );

    if (reply != QMessageBox::Yes) return;

    stats_bar_->setText(QString("⏳ Removing index for \"%1\" in progress...").arg(dir));
    IpcRequest req;
    req.action = IpcAction::UnindexDirectory;
    req.path = dir.toStdString();
    search_worker_->requestAction(req);
}

void MainWindow::onSearchCompleted(const IpcResponse& response) {
    if (response.action == IpcAction::IndexDirectory) {
        stats_bar_->setText(QString("✅ Indexing completed: %1 files indexed (%2 chunks) in %3 ms")
            .arg(response.total_files)
            .arg(response.total_chunks)
            .arg(response.latency_ms, 0, 'f', 1));
        QMessageBox::information(
            this,
            "Indexing Completed",
            QString("The directory was indexed successfully!\n\nFiles indexed: %1\nTotal chunks: %2\nTime elapsed: %3 ms")
                .arg(response.total_files)
                .arg(response.total_chunks)
                .arg(response.latency_ms, 0, 'f', 1)
        );
        return;
    }

    if (response.action == IpcAction::UnindexDirectory) {
        stats_bar_->setText(QString("🗑️ Index removed: %1 files deleted from index in %2 ms")
            .arg(response.total_files)
            .arg(response.latency_ms, 0, 'f', 1));
        QMessageBox::information(
            this,
            "Removal Completed",
            QString("Index removed successfully!\n\nFiles deleted from database: %1\nTime: %2 ms\n\nYou can now re-index the directory if needed.")
                .arg(response.total_files)
                .arg(response.latency_ms, 0, 'f', 1)
        );
        all_results_.clear();
        text_results_.clear();
        image_results_.clear();
        audio_results_.clear();
        btn_tab_text_->setText("📄 Text (0)");
        btn_tab_image_->setText("🖼️ Images (0)");
        btn_tab_audio_->setText("🎵 Audio (0)");
        renderResults();
        return;
    }

    if (response.action == IpcAction::ConfigureEngine) {
        if (!response.model_name.empty()) {
            label_model_name_->setText(QString::fromStdString(response.model_name));
        }
        stats_bar_->setText(QString("⚡ Engine configured: %1 (Device: %2, Latency: %3 ms)")
            .arg(label_model_name_->text())
            .arg(combo_device_->currentText())
            .arg(response.latency_ms, 0, 'f', 1));
        return;
    }

    if (response.action == IpcAction::UnindexFile) {
        stats_bar_->setText(QString("🗑️ File removed from search index in %1 ms")
            .arg(response.latency_ms, 0, 'f', 1));
        return;
    }

    raw_results_ = response.results;
    last_latency_ms_ = response.latency_ms;
    applyThresholdFilter();
}

void MainWindow::onSearchError(const QString& error_msg) {
    stats_bar_->setText(QString("⚠️ Search error: %1").arg(error_msg));
}

void MainWindow::onDaemonStatusChanged(bool is_running) {
    if (is_running) {
        daemon_status_badge_->setText("🟢 Daemon Connected");
        daemon_status_badge_->setStyleSheet("font-size: 11px; padding: 4px 10px; border-radius: 8px; background-color: #1e3a29; color: #a6da95; font-weight: bold;");
    } else {
        daemon_status_badge_->setText("🟠 Local Index");
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
            "No results to display.\n"
            "Start typing to search, or drag and drop an image or audio file into the window.",
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
            connect(card, &ResultCard::removeRequested, this, &MainWindow::onRemoveFileRequested);
            connect(card, &ResultCard::openFolderRequested, this, [](const SearchResult& res) {
                if (!res.filepath.empty()) {
                    QFileInfo fi(QString::fromStdString(res.filepath));
                    QString dir = fi.absolutePath();
                    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(dir))) {
                        QProcess::startDetached("xdg-open", {dir});
                    }
                }
            });
            results_layout_->addWidget(card);
        }
    };

    if (current_filter_ == DisplayFilter::AllGrouped) {
        // Grouped by Text, Images, and Audio
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
        search_input_->setText(QString("[Image: %1]").arg(fi.fileName()));
        active_query_mode_ = "Image (" + fi.fileName() + ")";
        IpcRequest req;
        req.action = IpcAction::SearchImage;
        req.path = file_path.toStdString();
        req.limit = 40;
        req.threshold = current_threshold_;
        stats_bar_->setText(QString("Searching for dropped image: %1...").arg(fi.fileName()));
        search_worker_->requestSearch(req);
    } else if (ext == "wav" || ext == "mp3" || ext == "flac" || ext == "ogg" || ext == "m4a") {
        search_input_->setText(QString("[Audio: %1]").arg(fi.fileName()));
        active_query_mode_ = "Audio (" + fi.fileName() + ")";
        IpcRequest req;
        req.action = IpcAction::SearchAudio;
        req.path = file_path.toStdString();
        req.limit = 40;
        req.threshold = current_threshold_;
        stats_bar_->setText(QString("Searching for dropped audio: %1...").arg(fi.fileName()));
        search_worker_->requestSearch(req);
    } else {
        // Assume text file
        search_input_->setText(fi.baseName());
        triggerSearch();
    }
}

void MainWindow::onDeviceChanged(int index) {
    bool use_gpu = combo_device_->itemData(index).toBool();
    search_worker_->setComputeDevice(use_gpu);
    stats_bar_->setText(QString("Compute device changed to %1.").arg(use_gpu ? "GPU (Vulkan)" : "CPU"));
}

void MainWindow::onChangeModel() {
    QString initial_dir = "models";
    if (!fs::exists("models")) {
        initial_dir = QDir::homePath();
    }
    QString file = QFileDialog::getOpenFileName(
        this,
        "Select GGUF Embedding Model",
        initial_dir,
        "GGUF Models (*.gguf);;All Files (*)"
    );
    if (!file.isEmpty()) {
        fs::path p = file.toStdString();
        label_model_name_->setText(QFileInfo(file).fileName());
        search_worker_->setModelPath(p);
        stats_bar_->setText(QString("Embedding model updating to: %1...").arg(QFileInfo(file).fileName()));
    }
}

void MainWindow::onRemoveFileRequested(const SearchResult& item) {
    auto reply = QMessageBox::question(
        this,
        "Remove File from Index",
        QString("Are you sure you want to remove \"%1\" from the search index?\n\nPath:\n%2")
            .arg(QString::fromStdString(item.filename))
            .arg(QString::fromStdString(item.filepath)),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No
    );

    if (reply != QMessageBox::Yes) {
        return;
    }

    std::string path_to_remove = item.filepath;
    IpcRequest req;
    req.action = IpcAction::UnindexFile;
    req.path = path_to_remove;
    search_worker_->requestAction(req);

    // Remove matching items from in-memory lists
    auto matches = [&](const SearchResult& r) { return r.filepath == path_to_remove; };
    raw_results_.erase(std::remove_if(raw_results_.begin(), raw_results_.end(), matches), raw_results_.end());
    all_results_.erase(std::remove_if(all_results_.begin(), all_results_.end(), matches), all_results_.end());
    text_results_.erase(std::remove_if(text_results_.begin(), text_results_.end(), matches), text_results_.end());
    image_results_.erase(std::remove_if(image_results_.begin(), image_results_.end(), matches), image_results_.end());
    audio_results_.erase(std::remove_if(audio_results_.begin(), audio_results_.end(), matches), audio_results_.end());

    // Update tab badges
    btn_tab_text_->setText(QString("📄 Text (%1)").arg(text_results_.size()));
    btn_tab_image_->setText(QString("🖼️ Images (%1)").arg(image_results_.size()));
    btn_tab_audio_->setText(QString("🎵 Audio (%1)").arg(audio_results_.size()));

    renderResults();
    stats_bar_->setText(QString("Removed \"%1\" from search index.").arg(QString::fromStdString(item.filename)));
}

void MainWindow::onThresholdChanged(int value) {
    current_threshold_ = static_cast<float>(value) / 100.0f;
    label_threshold_val_->setText(QString("%1%").arg(value));
    applyThresholdFilter();
}

void MainWindow::applyThresholdFilter() {
    all_results_.clear();
    text_results_.clear();
    image_results_.clear();
    audio_results_.clear();

    for (const auto& item : raw_results_) {
        Modality m = detectResultModality(item);
        if (m == Modality::Image || m == Modality::Audio) {
            if (current_threshold_ > 0.0f && item.score < current_threshold_) {
                continue;
            }
        } else {
            if (current_threshold_ > 0.0f) {
                if (item.similarity > 0.0f && item.similarity < current_threshold_ && item.rank_lexical == 0) {
                    continue;
                }
            }
        }

        all_results_.push_back(item);
        if (m == Modality::Text) {
            text_results_.push_back(item);
        } else if (m == Modality::Image) {
            image_results_.push_back(item);
        } else if (m == Modality::Audio) {
            audio_results_.push_back(item);
        }
    }

    // Update filter tab labels
    btn_tab_text_->setText(QString("📄 Text (%1)").arg(text_results_.size()));
    btn_tab_image_->setText(QString("🖼️ Images (%1)").arg(image_results_.size()));
    btn_tab_audio_->setText(QString("🎵 Audio (%1)").arg(audio_results_.size()));

    renderResults();

    stats_bar_->setText(QString("⚡ %1 results found in %2 ms | %3 (Threshold: %4%)")
        .arg(all_results_.size())
        .arg(last_latency_ms_, 0, 'f', 2)
        .arg(active_query_mode_)
        .arg(static_cast<int>(current_threshold_ * 100)));
}

} // namespace gemfinder
