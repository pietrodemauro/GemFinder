#include "gui/ResultCard.hpp"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDesktopServices>
#include <QUrl>
#include <QClipboard>
#include <QApplication>
#include <QFileInfo>
#include <QPixmap>
#include <QPainter>
#include <QPainterPath>
#include <QProcess>
#include <iomanip>
#include <sstream>

namespace gemfinder {

namespace {

QString formatOffset(const SearchResult& result) {
    if (result.chunk_type == "pdf_page") {
        if (result.start_offset > 0) {
            return QString("Page %1").arg(result.start_offset);
        }
        return QString("PDF");
    }
    Modality modality = detectResultModality(result);
    if (modality == Modality::Text) {
        if (result.start_offset > 0) {
            return QString("Line %1").arg(result.start_offset);
        }
        return QString("Text");
    }
    if (modality == Modality::Audio) {
        int64_t minutes = result.start_offset / 60;
        int64_t seconds = result.start_offset % 60;
        return QString("⏱ %1:%2")
            .arg(minutes, 2, 10, QChar('0'))
            .arg(seconds, 2, 10, QChar('0'));
    }
    return QString("🖼️ Frame");
}

QString formatScore(double score) {
    // Score in RRF is typically ~ 0.01 to 0.033
    // We can display both raw score or relative relevance
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4) << score;
    return QString::fromStdString(ss.str());
}

} // namespace

ResultCard::ResultCard(const SearchResult& result, QWidget* parent)
    : QFrame(parent), result_(result), modality_(detectResultModality(result)) {
    setupUi();
}

void ResultCard::setupUi() {
    setObjectName("ResultCard");
    setFrameShape(QFrame::StyledPanel);
    setCursor(Qt::PointingHandCursor);

    setStyleSheet(R"(
        #ResultCard {
            background-color: #24273a;
            border: 1px solid #363a4f;
            border-radius: 8px;
            padding: 8px;
            margin: 2px 4px;
        }
        #ResultCard:hover {
            background-color: #2c3047;
            border: 1px solid #89b4fa;
        }
    )");

    auto* main_layout = new QHBoxLayout(this);
    main_layout->setContentsMargins(10, 8, 10, 8);
    main_layout->setSpacing(12);

    // Left Icon/Thumbnail
    thumb_label_ = new QLabel(this);
    thumb_label_->setFixedSize(56, 56);
    thumb_label_->setAlignment(Qt::AlignCenter);
    thumb_label_->setStyleSheet("background-color: #1e2030; border-radius: 6px;");

    if (modality_ == Modality::Image && !result_.filepath.empty()) {
        QPixmap orig(QString::fromStdString(result_.filepath));
        if (!orig.isNull()) {
            QPixmap scaled = orig.scaled(56, 56, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
            // Center crop to 56x56
            QPixmap rounded(56, 56);
            rounded.fill(Qt::transparent);
            QPainter painter(&rounded);
            painter.setRenderHint(QPainter::Antialiasing);
            QPainterPath clip_path;
            clip_path.addRoundedRect(0, 0, 56, 56, 6, 6);
            painter.setClipPath(clip_path);
            int x = (scaled.width() - 56) / 2;
            int y = (scaled.height() - 56) / 2;
            painter.drawPixmap(0, 0, scaled, x, y, 56, 56);
            thumb_label_->setPixmap(rounded);
        } else {
            thumb_label_->setText("🖼️");
            thumb_label_->setStyleSheet("font-size: 26px; background-color: #1e2030; border-radius: 6px;");
        }
    } else if (modality_ == Modality::Audio) {
        thumb_label_->setText("🎵");
        thumb_label_->setStyleSheet("font-size: 26px; background-color: #27233a; border: 1px solid #b4befe; border-radius: 6px;");
    } else {
        thumb_label_->setText("📄");
        thumb_label_->setStyleSheet("font-size: 26px; background-color: #1f2d3d; border: 1px solid #89dceb; border-radius: 6px;");
    }
    main_layout->addWidget(thumb_label_);

    // Center Details
    auto* details_layout = new QVBoxLayout();
    details_layout->setSpacing(3);

    // Header row: filename + badges
    auto* header_layout = new QHBoxLayout();
    header_layout->setSpacing(8);

    title_label_ = new QLabel(QString::fromStdString(result_.filename), this);
    title_label_->setStyleSheet("color: #cad3f5; font-weight: bold; font-size: 14px;");
    title_label_->setCursor(Qt::PointingHandCursor);
    title_label_->setToolTip("Click to open containing folder");
    header_layout->addWidget(title_label_);

    offset_badge_ = new QLabel(formatOffset(result_), this);
    offset_badge_->setStyleSheet("background-color: #363a4f; color: #a6da95; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;");
    header_layout->addWidget(offset_badge_);

    QString score_str;
    if (modality_ == Modality::Image || modality_ == Modality::Audio) {
        score_str = QString("Sim: %1%").arg(static_cast<int>(std::round(result_.score * 100.0)));
    } else {
        if (result_.similarity > 0.01f) {
            score_str = QString("Sim: %1% | RRF: %2").arg(static_cast<int>(std::round(result_.similarity * 100.0f))).arg(formatScore(result_.score));
        } else {
            score_str = QString("RRF: %1").arg(formatScore(result_.score));
        }
    }
    score_badge_ = new QLabel(score_str, this);
    score_badge_->setStyleSheet("background-color: #494d64; color: #f5a97f; padding: 2px 6px; border-radius: 4px; font-size: 11px;");
    header_layout->addWidget(score_badge_);

    if (result_.modified_on_disk) {
        auto* mod_badge = new QLabel("⚠️ Modified on disk", this);
        mod_badge->setStyleSheet("background-color: #f5a97f; color: #1e2030; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;");
        mod_badge->setToolTip("This file was modified on disk since it was indexed. Click 'Re-index' to update.");
        header_layout->addWidget(mod_badge);
    }

    header_layout->addStretch();
    details_layout->addLayout(header_layout);

    // Path row
    path_label_ = new QLabel(QString::fromStdString(result_.filepath), this);
    path_label_->setStyleSheet("color: #8087a2; font-size: 11px;");
    path_label_->setCursor(Qt::PointingHandCursor);
    path_label_->setToolTip("Click to open containing folder: " + QString::fromStdString(result_.filepath));
    details_layout->addWidget(path_label_);

    // Snippet preview row
    QString preview_str = QString::fromStdString(result_.preview_text).trimmed();
    if (preview_str.isEmpty()) {
        if (modality_ == Modality::Image) {
            preview_str = "[Image embedded and indexed]";
        } else if (modality_ == Modality::Audio) {
            preview_str = "[Audio segment sampled at 16 kHz Mono]";
        }
    }
    // Limit length for display
    if (preview_str.length() > 180) {
        preview_str = preview_str.left(177) + "...";
    }

    preview_label_ = new QLabel(preview_str, this);
    preview_label_->setStyleSheet("color: #a5adcb; font-size: 12px; font-style: italic;");
    preview_label_->setWordWrap(true);
    details_layout->addWidget(preview_label_);

    main_layout->addLayout(details_layout, 1);

    // Action buttons on the right
    auto* actions_layout = new QVBoxLayout();
    actions_layout->setSpacing(4);
    actions_layout->setAlignment(Qt::AlignCenter);

    btn_open_folder_ = new QPushButton("📁 Folder", this);
    btn_open_folder_->setToolTip("Open containing folder in file manager");
    btn_open_folder_->setCursor(Qt::PointingHandCursor);
    btn_open_folder_->setStyleSheet(
        "QPushButton { background-color: #363a4f; color: #89b4fa; border: 1px solid #494d64; border-radius: 6px; padding: 4px 10px; font-size: 11px; font-weight: bold; } "
        "QPushButton:hover { background-color: #89b4fa; color: #1e2030; }"
    );
    connect(btn_open_folder_, &QPushButton::clicked, this, &ResultCard::openContainingFolder);
    actions_layout->addWidget(btn_open_folder_);

    btn_open_file_ = new QPushButton("📄 Open", this);
    btn_open_file_->setToolTip("Open file with default system application");
    btn_open_file_->setCursor(Qt::PointingHandCursor);
    btn_open_file_->setStyleSheet(
        "QPushButton { background-color: #363a4f; color: #a6da95; border: 1px solid #494d64; border-radius: 6px; padding: 4px 10px; font-size: 11px; } "
        "QPushButton:hover { background-color: #a6da95; color: #1e2030; }"
    );
    connect(btn_open_file_, &QPushButton::clicked, this, &ResultCard::openFile);
    actions_layout->addWidget(btn_open_file_);

    btn_reindex_ = new QPushButton("🔄 Re-index", this);
    btn_reindex_->setToolTip("Re-index this file into the search index");
    btn_reindex_->setCursor(Qt::PointingHandCursor);
    btn_reindex_->setStyleSheet(
        "QPushButton { background-color: #363a4f; color: #eed49f; border: 1px solid #494d64; border-radius: 6px; padding: 4px 10px; font-size: 11px; } "
        "QPushButton:hover { background-color: #eed49f; color: #1e2030; }"
    );
    connect(btn_reindex_, &QPushButton::clicked, this, [this]() {
        emit reindexRequested(result_);
    });
    actions_layout->addWidget(btn_reindex_);

    btn_remove_ = new QPushButton("🗑️ Remove", this);
    btn_remove_->setToolTip("Remove this file from the search index");
    btn_remove_->setCursor(Qt::PointingHandCursor);
    btn_remove_->setStyleSheet(
        "QPushButton { background-color: #363a4f; color: #ed8796; border: 1px solid #494d64; border-radius: 6px; padding: 4px 10px; font-size: 11px; } "
        "QPushButton:hover { background-color: #ed8796; color: #1e2030; }"
    );
    connect(btn_remove_, &QPushButton::clicked, this, [this]() {
        emit removeRequested(result_);
    });
    actions_layout->addWidget(btn_remove_);

    main_layout->addLayout(actions_layout);
}

void ResultCard::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        emit cardClicked(result_);
        emit openFolderRequested(result_);
        openContainingFolder();
    }
    QFrame::mousePressEvent(event);
}

void ResultCard::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        emit cardDoubleClicked(result_);
        openFile();
    }
    QFrame::mouseDoubleClickEvent(event);
}

void ResultCard::contextMenuEvent(QContextMenuEvent* event) {
    QMenu menu(this);
    menu.setStyleSheet(R"(
        QMenu {
            background-color: #1e2030;
            color: #cad3f5;
            border: 1px solid #363a4f;
            border-radius: 6px;
            padding: 4px;
        }
        QMenu::item {
            padding: 6px 20px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background-color: #363a4f;
            color: #89b4fa;
        }
    )");

    auto* act_folder = menu.addAction("📁 Show in Folder");
    auto* act_open = menu.addAction("📂 Open File");
    auto* act_reindex = menu.addAction("🔄 Re-index File");
    auto* act_remove = menu.addAction("🗑️ Remove from Index");
    menu.addSeparator();
    auto* act_copy_path = menu.addAction("📋 Copy Absolute Path");
    auto* act_copy_preview = menu.addAction("📝 Copy Preview Text");

    QAction* selected = menu.exec(event->globalPos());
    if (selected == act_folder) {
        openContainingFolder();
    } else if (selected == act_open) {
        openFile();
    } else if (selected == act_reindex) {
        emit reindexRequested(result_);
    } else if (selected == act_remove) {
        emit removeRequested(result_);
    } else if (selected == act_copy_path) {
        copyPath();
    } else if (selected == act_copy_preview) {
        copyPreviewText();
    }
}

void ResultCard::openFile() {
    if (!result_.filepath.empty()) {
        QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdString(result_.filepath)));
    }
}

void ResultCard::openContainingFolder() {
    if (!result_.filepath.empty()) {
        QFileInfo fi(QString::fromStdString(result_.filepath));
        QString dir_path = fi.absolutePath();
        if (!QDesktopServices::openUrl(QUrl::fromLocalFile(dir_path))) {
            QProcess::startDetached("xdg-open", {dir_path});
        }
    }
}

void ResultCard::copyPath() {
    if (!result_.filepath.empty()) {
        QClipboard* cb = QApplication::clipboard();
        cb->setText(QString::fromStdString(result_.filepath));
    }
}

void ResultCard::copyPreviewText() {
    if (!result_.preview_text.empty()) {
        QClipboard* cb = QApplication::clipboard();
        cb->setText(QString::fromStdString(result_.preview_text));
    }
}

} // namespace gemfinder
