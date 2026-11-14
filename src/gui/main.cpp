#include "gui/MainWindow.hpp"

#include <QApplication>
#include <QFont>
#include <QStyleFactory>
#include <iostream>
#include <optional>
#include <filesystem>
#include <string_view>

int main(int argc, char* argv[]) {
    std::optional<std::filesystem::path> model_path = std::nullopt;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    // Enable High DPI scaling
    QApplication app(argc, argv);
    app.setApplicationName("GemFinder");
    app.setOrganizationName("GemFinder");
    app.setApplicationVersion("0.1.0");

    // Modern Fusion style base
    app.setStyle(QStyleFactory::create("Fusion"));

    QFont font = app.font();
    font.setPointSize(10);
    app.setFont(font);

    gemfinder::MainWindow window(model_path);
    window.show();

    return app.exec();
}
