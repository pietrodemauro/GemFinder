#include "gui/MainWindow.hpp"

#include <QApplication>
#include <QFont>
#include <QStyleFactory>
#include <iostream>

int main(int argc, char* argv[]) {
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

    gemfinder::MainWindow window;
    window.show();

    return app.exec();
}
