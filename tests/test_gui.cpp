#ifdef NDEBUG
#undef NDEBUG
#endif

#include "gui/SearchResultItem.hpp"
#include "gui/CategoryHeader.hpp"
#include "gui/ResultCard.hpp"
#include "gui/MainWindow.hpp"

#include <QApplication>
#include <QTimer>
#include <cassert>
#include <iostream>

using namespace gemfinder;

void testModalityDetection() {
    std::cout << "[Test] testModalityDetection..." << std::endl;

    SearchResult r1;
    r1.chunk_type = "text_paragraph";
    r1.filepath = "/tmp/sample.txt";
    assert(detectResultModality(r1) == Modality::Text);

    SearchResult r2;
    r2.chunk_type = "image_frame";
    r2.filepath = "/tmp/photo.jpg";
    assert(detectResultModality(r2) == Modality::Image);

    SearchResult r3;
    r3.chunk_type = "audio_segment";
    r3.filepath = "/tmp/song.wav";
    assert(detectResultModality(r3) == Modality::Audio);

    SearchResult r_pdf;
    r_pdf.chunk_type = "pdf_page";
    r_pdf.filepath = "/tmp/report.pdf";
    r_pdf.start_offset = 3;
    assert(detectResultModality(r_pdf) == Modality::Text);

    // Fallback detection via extension
    SearchResult r4;
    r4.chunk_type = "";
    r4.filepath = "/tmp/screenshot.png";
    assert(detectResultModality(r4) == Modality::Image);

    SearchResult r5;
    r5.chunk_type = "";
    r5.filepath = "/tmp/voice.mp3";
    assert(detectResultModality(r5) == Modality::Audio);

    SearchResult r6;
    r6.chunk_type = "";
    r6.filepath = "/tmp/main.cpp";
    assert(detectResultModality(r6) == Modality::Text);

    std::cout << "  -> Modality detection passed." << std::endl;
}

void testGroupingLogic() {
    std::cout << "[Test] testGroupingLogic..." << std::endl;

    std::vector<SearchResult> results;

    SearchResult t1;
    t1.filename = "doc1.txt";
    t1.filepath = "/path/doc1.txt";
    t1.chunk_type = "text_paragraph";
    t1.score = 0.032;
    results.push_back(t1);

    SearchResult i1;
    i1.filename = "pic1.png";
    i1.filepath = "/path/pic1.png";
    i1.chunk_type = "image_frame";
    i1.score = 0.028;
    results.push_back(i1);

    SearchResult a1;
    a1.filename = "audio1.wav";
    a1.filepath = "/path/audio1.wav";
    a1.chunk_type = "audio_segment";
    a1.score = 0.025;
    results.push_back(a1);

    SearchResult t2;
    t2.filename = "doc2.md";
    t2.filepath = "/path/doc2.md";
    t2.chunk_type = "text_paragraph";
    t2.score = 0.021;
    results.push_back(t2);

    std::vector<SearchResult> text_items;
    std::vector<SearchResult> image_items;
    std::vector<SearchResult> audio_items;

    for (const auto& item : results) {
        Modality m = detectResultModality(item);
        if (m == Modality::Text) text_items.push_back(item);
        else if (m == Modality::Image) image_items.push_back(item);
        else if (m == Modality::Audio) audio_items.push_back(item);
    }

    assert(text_items.size() == 2);
    assert(image_items.size() == 1);
    assert(audio_items.size() == 1);

    assert(text_items[0].filename == "doc1.txt");
    assert(text_items[1].filename == "doc2.md");
    assert(image_items[0].filename == "pic1.png");
    assert(audio_items[0].filename == "audio1.wav");

    std::cout << "  -> Result grouping logic passed." << std::endl;
}

void testWidgetsInstantiation(int argc, char** argv) {
    std::cout << "[Test] testWidgetsInstantiation..." << std::endl;

    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);

    // Test CategoryHeader
    CategoryHeader header_text(Modality::Text, 5);
    header_text.setCount(10);

    CategoryHeader header_img(Modality::Image, 2);
    CategoryHeader header_aud(Modality::Audio, 7);

    // Test ResultCard
    SearchResult item;
    item.filename = "example.cpp";
    item.filepath = "/tmp/example.cpp";
    item.preview_text = "int main() { return 0; }";
    item.chunk_type = "text_paragraph";
    item.start_offset = 42;
    item.score = 0.033;

    ResultCard card(item);
    assert(card.result().filename == "example.cpp");
    assert(card.modality() == Modality::Text);

    // Test MainWindow instantiation
    MainWindow window;
    window.show();

    // Close after 50ms
    QTimer::singleShot(50, &app, &QApplication::quit);
    app.exec();

    std::cout << "  -> Widgets instantiation passed." << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== GemFinder GUI Test Suite ===" << std::endl;

    testModalityDetection();
    testGroupingLogic();
    testWidgetsInstantiation(argc, argv);

    std::cout << "All GUI tests passed successfully!" << std::endl;
    return 0;
}
