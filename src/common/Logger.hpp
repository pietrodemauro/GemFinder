#pragma once

#include <iostream>
#include <string_view>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace gemfinder {

enum class LogLevel {
    Debug,
    Info,
    Warn,
    Error
};

class Logger {
public:
    static void log(LogLevel level, std::string_view msg) {
        const auto now = std::chrono::system_clock::now();
        const auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf{};
        localtime_r(&time_t_now, &tm_buf);

        std::ostream& os = (level == LogLevel::Error) ? std::cerr : std::cout;

        os << "[" << std::put_time(&tm_buf, "%H:%M:%S") << "] ";

        switch (level) {
            case LogLevel::Debug:
                os << "\033[36m[DEBUG]\033[0m ";
                break;
            case LogLevel::Info:
                os << "\033[32m[INFO] \033[0m ";
                break;
            case LogLevel::Warn:
                os << "\033[33m[WARN] \033[0m ";
                break;
            case LogLevel::Error:
                os << "\033[31m[ERROR]\033[0m ";
                break;
        }

        os << msg << "\n";
    }

    static void info(std::string_view msg)  { log(LogLevel::Info, msg); }
    static void warn(std::string_view msg)  { log(LogLevel::Warn, msg); }
    static void error(std::string_view msg) { log(LogLevel::Error, msg); }
    static void debug(std::string_view msg) { log(LogLevel::Debug, msg); }
};

} // namespace gemfinder
