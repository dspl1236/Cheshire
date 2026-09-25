// cheshire/env.h - one reading of the CHESHIRE_* environment variables for every port.
//
// Before this header, the ports and the generator's inline code parsed their switches five different
// ways: `getenv(X) != nullptr` (so `X=0` turned a check ON), `e[0] == '0'`, `e[0] == '1'` (so
// `X=true` did nothing), atoi / strtoll / strtod (so `X=abc` meant 0), and string compares. Every
// CHESHIRE_* read now goes through these functions, with one rule per kind:
//
//   flag     unset -> the default; "0", "false", "off", "no" (any case) or empty -> off; else on
//   integer  unset, empty or not a number -> the default; otherwise strtoll's value
//   real     the same with strtod
//   text     unset -> the default string; otherwise the value as given
//   isSet    set at all, to anything (for a value that is a path)
//
// Header-only and free of AliceVision types, so the same file serves host C++, HIP and nvcc
// translation units. Callers that read a switch in a hot path keep doing so once, in a
// function-local static, as before. scripts/apply_hip_patch.py copies this file next to bridge.h and
// refuses a generated tree that still reads a CHESHIRE_* variable any other way.
#pragma once

#include <cctype>
#include <cstdlib>
#include <string>

namespace cheshire {
namespace env {
namespace detail {
// MSVC's C4996 on getenv, once here instead of at every former call site: each value is parsed or
// copied at once and the pointer never kept, which is the case the warning is about.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
inline const char* raw(const char* name) { return std::getenv(name); }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}  // namespace detail

inline bool isSet(const char* name) { return detail::raw(name) != nullptr; }

inline bool flag(const char* name, bool def = false)
{
    const char* v = detail::raw(name);
    if (v == nullptr)
        return def;
    std::string s;
    for (const char* p = v; *p; ++p)
        if (!std::isspace(static_cast<unsigned char>(*p)))
            s += static_cast<char>(std::tolower(static_cast<unsigned char>(*p)));
    return !(s.empty() || s == "0" || s == "false" || s == "off" || s == "no");
}

inline long long integer(const char* name, long long def)
{
    const char* v = detail::raw(name);
    if (v == nullptr || *v == '\0')
        return def;
    char* end = nullptr;
    const long long x = std::strtoll(v, &end, 10);
    return end == v ? def : x;
}

inline double real(const char* name, double def)
{
    const char* v = detail::raw(name);
    if (v == nullptr || *v == '\0')
        return def;
    char* end = nullptr;
    const double x = std::strtod(v, &end);
    return end == v ? def : x;
}

inline std::string text(const char* name, const char* def = "")
{
    const char* v = detail::raw(name);
    return v != nullptr ? std::string(v) : std::string(def);
}

}  // namespace env
}  // namespace cheshire
