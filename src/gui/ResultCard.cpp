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
#include <iomanip>
#include <sstream>

namespace gemfinder {

namespace {

QString formatOffset(Modality modality, int64_t offset) {
    if (modality == Modality::Text) {
        if (offset > 0) {
            return QString("Riga %1").arg(offset);
        }
        return QString("Testo");
    }
    if (modality == Modality::Audio) {
        int64_t minutes = offset / 60;
        int64_t seconds = offset % 60;
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
    header_layout->addWidget(title_label_);

    offset_badge_ = new QLabel(formatOffset(modality_, result_.start_offset), this);
    offset_badge_->setStyleSheet("background-color: #363a4f; color: #a6da95; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;");
    header_layout->addWidget(offset_badge_);

    score_badge_ = new QLabel(QString("RRF: %1").arg(formatScore(result_.score)), this);
    score_badge_->setStyleSheet("background-color: #494d64; color: #f5a97f; padding: 2px 6px; border-radius: 4px; font-size: 11px;");
    header_layout->addWidget(score_badge_);

    header_layout->addStretch();
    details_layout->addLayout(header_layout);

    // Path row
    path_label_ = new QLabel(QString::fromStdString(result_.filepath), this);
    path_label_->setStyleSheet("color: #8087a2; font-size: 11px;");
    path_label_->setToolTip(QString::fromStdString(result_.filepath));
    details_layout->addWidget(path_label_);

    // Snippet preview row
    QString preview_str = QString::fromStdString(result_.preview_text).trimmed();
    if (preview_str.isEmpty()) {
        if (modality_ == Modality::Image) {
            preview_str = "[Immagine vettorializzata e indicizzata]";
        } else if (modality_ == Modality::Audio) {
            preview_str = "[Segmento audio campionato a 16 kHz Mono]";
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
}

void ResultCard::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        emit cardClicked(result_);
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

    auto* act_open = menu.addAction("📂 Apri File");
    auto* act_folder = menu.addAction("📁 Mostra nella Cartella");
    menu.addSeparator();
    auto* act_copy_path = menu.addAction("📋 Copia Percorso Assoluto");
    auto* act_copy_preview = menu.addAction("📝 Copia Testo Anteprima");

    QAction* selected = menu.exec(event->globalPos());
    if (selected == act_open) {
        openFile();
    } else if (selected == act_folder) {
        openContainingFolder();
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
        QDesktopServices::openUrl(QUrl::fromLocalFile(fi.absolutePath()));
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
