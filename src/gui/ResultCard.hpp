#pragma once

#include "common/Types.hpp"
#include "gui/SearchResultItem.hpp"
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QMouseEvent>
#include <QMenu>

namespace gemfinder {

class ResultCard : public QFrame {
    Q_OBJECT

public:
    explicit ResultCard(const SearchResult& result, QWidget* parent = nullptr);
    ~ResultCard() override = default;

    [[nodiscard]] const SearchResult& result() const noexcept { return result_; }
    [[nodiscard]] Modality modality() const noexcept { return modality_; }

signals:
    void cardClicked(const SearchResult& result);
    void cardDoubleClicked(const SearchResult& result);
    void removeRequested(const SearchResult& result);
    void reindexRequested(const SearchResult& result);
    void openFolderRequested(const SearchResult& result);

public slots:
    void openFile();
    void openContainingFolder();
    void copyPath();
    void copyPreviewText();

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override;

private:
    void setupUi();

    SearchResult result_;
    Modality modality_;
    QLabel* thumb_label_{nullptr};
    QLabel* title_label_{nullptr};
    QLabel* path_label_{nullptr};
    QLabel* preview_label_{nullptr};
    QLabel* score_badge_{nullptr};
    QLabel* offset_badge_{nullptr};
    QPushButton* btn_open_folder_{nullptr};
    QPushButton* btn_open_file_{nullptr};
    QPushButton* btn_reindex_{nullptr};
    QPushButton* btn_remove_{nullptr};
};

} // namespace gemfinder
