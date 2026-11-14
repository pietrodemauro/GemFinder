#pragma once

#include "common/Types.hpp"
#include <QFrame>
#include <QLabel>
#include <QString>

namespace gemfinder {

class CategoryHeader : public QFrame {
    Q_OBJECT

public:
    explicit CategoryHeader(Modality modality, size_t count = 0, QWidget* parent = nullptr);
    ~CategoryHeader() override = default;

    void setCount(size_t count);

private:
    void setupUi();

    Modality modality_;
    size_t count_{0};
    QLabel* title_label_{nullptr};
    QLabel* count_badge_{nullptr};
};

} // namespace gemfinder
