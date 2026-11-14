#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include "daemon/Debouncer.hpp"
#include "daemon/FileWatcher.hpp"
#include "daemon/IpcServer.hpp"
#include "daemon/IpcClient.hpp"
#include "common/Logger.hpp"

#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

void testDebouncer() {
    gemfinder::Logger::info("--- Test 1: Debouncer Coalescing ---");
    gemfinder::Debouncer debouncer(std::chrono::milliseconds(50));

    fs::path p1 = "/tmp/file1.txt";
    debouncer.addEvent(p1, gemfinder::FileEvent::Created);
    debouncer.addEvent(p1, gemfinder::FileEvent::Modified);
    debouncer.addEvent(p1, gemfinder::FileEvent::Modified);

    // Immediately poll -> should be empty (not expired yet)
    auto expired_early = debouncer.pollExpired();
    assert(expired_early.empty());
    assert(debouncer.pendingCount() == 1);

    // Sleep 70ms -> should expire as 1 coalesced event
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    auto expired = debouncer.pollExpired();
    assert(expired.size() == 1);
    assert(expired[0].first == p1);
    assert(expired[0].second == gemfinder::FileEvent::Modified);
    assert(debouncer.pendingCount() == 0);

    gemfinder::Logger::info("Test 1 completato con successo!");
}

void testIpcServerAndClient() {
    gemfinder::Logger::info("--- Test 2: IPC Server & Client Roundtrip ---");
    fs::path sock_path = "/tmp/test_ipc_roundtrip.sock";

    gemfinder::IpcServer server(sock_path, [](const gemfinder::IpcRequest& req) -> gemfinder::IpcResponse {
        gemfinder::IpcResponse resp;
        resp.success = true;
        if (req.action == gemfinder::IpcAction::Ping) {
            resp.latency_ms = 0.05;
        } else if (req.action == gemfinder::IpcAction::SearchText) {
            gemfinder::SearchResult r;
            r.filename = "risultato_test.txt";
            r.filepath = "/tmp/risultato_test.txt";
            r.score = 0.95;
            r.preview_text = "Anteprima del testo trovato via IPC";
            resp.results.push_back(r);
            resp.latency_ms = 0.25;
        }
        return resp;
    });

    assert(server.start());

    gemfinder::IpcClient client(sock_path);
    assert(client.isDaemonRunning());

    gemfinder::IpcRequest req;
    req.action = gemfinder::IpcAction::SearchText;
    req.query = "test semantico";
    req.limit = 5;

    auto resp_opt = client.send(req);
    assert(resp_opt.has_value());
    assert(resp_opt->success);
    assert(resp_opt->results.size() == 1);
    assert(resp_opt->results[0].filename == "risultato_test.txt");

    gemfinder::Logger::info("IPC Roundtrip OK: Latency=" + std::to_string(resp_opt->latency_ms) + " ms, File=" + resp_opt->results[0].filename);

    server.stop();
    assert(!client.isDaemonRunning());

    gemfinder::Logger::info("Test 2 completato con successo!");
}

void testFileWatcher() {
    gemfinder::Logger::info("--- Test 3: inotify FileWatcher ---");
    fs::path test_dir = "/tmp/test_gemfinder_watcher";
    fs::remove_all(test_dir);
    fs::create_directories(test_dir);

    std::atomic<int> events_received{0};
    gemfinder::FileWatcher watcher([&](const fs::path& path, gemfinder::FileEvent ev) {
        (void)path;
        (void)ev;
        events_received++;
    }, std::chrono::milliseconds(50));

    assert(watcher.start());
    assert(watcher.addWatch(test_dir));

    // Create file
    fs::path f = test_dir / "nuovo_file.txt";
    {
        std::ofstream out(f);
        out << "Contenuto del file monitorato";
    }

    // Wait for inotify + debouncer
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    assert(events_received.load() >= 1);
    gemfinder::Logger::info("FileWatcher rilevato evento con successo (event count: " + std::to_string(events_received.load()) + ")");

    watcher.stop();
    fs::remove_all(test_dir);
    gemfinder::Logger::info("Test 3 completato con successo!");
}

int main() {
    gemfinder::Logger::info("=== Avvio Test Suite Daemon, inotify Watcher & IPC (Step 5) ===");
    testDebouncer();
    testIpcServerAndClient();
    testFileWatcher();
    gemfinder::Logger::info("=== TUTTI I TEST DELLO STEP 5 SUPERATI CON SUCCESSO! ===");
    return 0;
}
