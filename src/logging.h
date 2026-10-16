// Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Original utility; no adapted code.
#pragma once
#include <cstdlib>
#include <iostream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
enum class LogLevel { Success, Error, Warn, Process };
inline void init_logging() {
#ifdef _WIN32
    for (DWORD id : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE}) {
        HANDLE handle = GetStdHandle(id);
        DWORD mode;
        if (GetConsoleMode(handle, &mode))
            SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
#endif
}
inline std::ostream& log_message(LogLevel level) {
    const char* label = "[PROCESS]:";
    const char* color = "\033[37m";
    if (level == LogLevel::Success) { label = "[SUCCESS]:"; color = "\033[32m"; }
    if (level == LogLevel::Error) { label = "[ERROR]:"; color = "\033[31m"; }
    if (level == LogLevel::Warn) { label = "[WARN]:"; color = "\033[33m"; }
    auto& stream = level == LogLevel::Error ? std::cerr : std::cout;
    if (std::getenv("NO_COLOR")) return stream << label << ' ';
    return stream << color << label << "\033[0m ";
}
