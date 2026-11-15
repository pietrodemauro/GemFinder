#pragma once

#include "common/Types.hpp"
#include "gui/SearchWorker.hpp"
#include "gui/ResultCard.hpp"
#include "gui/CategoryHeader.hpp"

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTimer>
#include <QButtonGroup>
#include <QComboBox>
#include <QSlider>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <vector>

namespace gemfinder {

enum class DisplayFilter {
    AllGrouped,
    TextOnly,
    ImageOnly,
    AudioOnly
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(std::optional<fs::path> model_path = std::nullopt, int n_threads = 0, QWidget* parent = nullptr);
    ~MainWindow() override;

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private slots:
    void onSearchTextChanged(const QString& text);
    void onDebounceTimeout();
    void onSearchCompleted(const IpcResponse& response);
    void onSearchError(const QString& error_msg);
    void onDaemonStatusChanged(bool is_running);
    void onSelectImageQuery();
    void onSelectAudioQuery();
    void onIndexDirectory();
    void onUnindexDirectory();
    void onFilterChanged(DisplayFilter filter);
    void onDeviceChanged(int index);
    void onChangeModel();
    void onRemoveFileRequested(const SearchResult& result);
    void onThresholdChanged(int value);

private:
    void setupUi();
    void setupStyles();
    void triggerSearch();
    void renderResults();
    void clearResultsLayout();
    void applyThresholdFilter();

    // UI elements
    QLineEdit* search_input_{nullptr};
    QPushButton* btn_image_search_{nullptr};
    QPushButton* btn_audio_search_{nullptr};
    QPushButton* btn_index_dir_{nullptr};
    QPushButton* btn_unindex_dir_{nullptr};
    QLabel* daemon_status_badge_{nullptr};
    QLabel* stats_bar_{nullptr};

    // Hardware & Model Controls
    QComboBox* combo_device_{nullptr};
    QLabel* label_model_name_{nullptr};
    QPushButton* btn_change_model_{nullptr};
    QSlider* slider_threshold_{nullptr};
    QLabel* label_threshold_val_{nullptr};
    float current_threshold_{0.0f};

    // Filter Buttons
    QPushButton* btn_tab_all_{nullptr};
    QPushButton* btn_tab_text_{nullptr};
    QPushButton* btn_tab_image_{nullptr};
    QPushButton* btn_tab_audio_{nullptr};

    // Results container
    QScrollArea* scroll_area_{nullptr};
    QWidget* results_container_{nullptr};
    QVBoxLayout* results_layout_{nullptr};

    // Worker & State
    SearchWorker* search_worker_{nullptr};
    QTimer* debounce_timer_{nullptr};
    DisplayFilter current_filter_{DisplayFilter::AllGrouped};

    // Stored results
    std::vector<SearchResult> raw_results_;
    std::vector<SearchResult> all_results_;
    std::vector<SearchResult> text_results_;
    std::vector<SearchResult> image_results_;
    std::vector<SearchResult> audio_results_;

    double last_latency_ms_{0.0};
    QString active_query_mode_{"Text"};
};

} // namespace gemfinder
