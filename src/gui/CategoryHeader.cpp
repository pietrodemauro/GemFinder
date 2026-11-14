#include "gui/CategoryHeader.hpp"

#include <QHBoxLayout>

namespace gemfinder {

CategoryHeader::CategoryHeader(Modality modality, size_t count, QWidget* parent)
    : QFrame(parent), modality_(modality), count_(count) {
    setupUi();
}

void CategoryHeader::setupUi() {
    setObjectName("CategoryHeader");
    setStyleSheet(R"(
        #CategoryHeader {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #181926, stop:1 #24273a);
            border-left: 3px solid #89b4fa;
            border-radius: 4px;
            margin-top: 10px;
            margin-bottom: 4px;
        }
    )");

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(12, 6, 12, 6);
    layout->setSpacing(8);

    QString title;
    QString icon;
    QString border_color;
    if (modality_ == Modality::Text) {
        icon = "📄";
        title = "DOCUMENTI & TESTO";
        border_color = "#89b4fa";
    } else if (modality_ == Modality::Image) {
        icon = "🖼️";
        title = "IMMAGINI";
        border_color = "#f5bde6";
    } else {
        icon = "🎵";
        title = "AUDIO & REGISTRAZIONI";
        border_color = "#a6da95";
    }

    setStyleSheet(QString(R"(
        #CategoryHeader {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #181926, stop:1 #24273a);
            border-left: 4px solid %1;
            border-radius: 4px;
            margin-top: 10px;
            margin-bottom: 4px;
        }
    )").arg(border_color));

    title_label_ = new QLabel(QString("%1  %2").arg(icon, title), this);
    title_label_->setStyleSheet("color: #eed49f; font-weight: bold; font-size: 13px; letter-spacing: 1px;");
    layout->addWidget(title_label_);

    layout->addStretch();

    count_badge_ = new QLabel(QString("%1 risultati").arg(count_), this);
    count_badge_->setStyleSheet("background-color: #363a4f; color: #cad3f5; font-size: 11px; padding: 2px 8px; border-radius: 10px; font-weight: bold;");
    layout->addWidget(count_badge_);
}

void CategoryHeader::setCount(size_t count) {
    count_ = count;
    if (count_badge_) {
        count_badge_->setText(QString("%1 risultati").arg(count_));
    }
}

} // namespace gemfinder
