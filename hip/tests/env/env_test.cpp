// cheshire/env.h: every rule, on the host. Build and run:
//   cl /nologo /EHsc /std:c++17 /I ..\..\compat\include env_test.cpp && env_test.exe
//   g++ -std=c++17 -I ../../compat/include env_test.cpp -o env_test && ./env_test
#include <cheshire/env.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _WIN32
static void put(const char* name, const char* value) { _putenv_s(name, value ? value : ""); }
static bool canSetEmpty() { return false; }  // _putenv_s(name, "") removes the variable
#else
static void put(const char* name, const char* value)
{
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
}
static bool canSetEmpty() { return true; }
#endif

static int failures = 0;
static void expect(bool ok, const char* what)
{
    if (!ok)
    {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

int main()
{
    using namespace cheshire::env;
    const char* N = "CHESHIRE_ENV_TEST";

    put(N, nullptr);
    expect(!isSet(N), "unset: isSet false");
    expect(!flag(N), "unset flag: default false");
    expect(flag(N, true), "unset flag: default true");
    expect(integer(N, 7) == 7, "unset integer: default");
    expect(real(N, 2.5) == 2.5, "unset real: default");
    expect(text(N, "dflt") == "dflt", "unset text: default");

    for (const char* off : {"0", "false", "FALSE", "Off", "no", " 0 "})
    {
        put(N, off);
        expect(isSet(N), "set: isSet");
        expect(!flag(N, true), (std::string("off value: ") + off).c_str());
    }
    for (const char* on : {"1", "true", "yes", "on", "2", "check", "anything"})
    {
        put(N, on);
        expect(flag(N, false), (std::string("on value: ") + on).c_str());
    }
    if (canSetEmpty())
    {
        put(N, "");
        expect(isSet(N), "empty: isSet");
        expect(!flag(N, true), "empty flag: off");
        expect(integer(N, 9) == 9, "empty integer: default");
        expect(text(N, "d").empty(), "empty text: empty, not the default");
    }

    put(N, "512");
    expect(integer(N, 1) == 512, "integer 512");
    put(N, "-3");
    expect(integer(N, 1) == -3, "integer -3");
    put(N, "64MB");
    expect(integer(N, 1) == 64, "integer with a unit suffix: the leading number, as strtoll");
    put(N, "abc");
    expect(integer(N, 11) == 11, "unparsable integer: default");
    put(N, "0.25");
    expect(real(N, 1.0) == 0.25, "real 0.25");
    put(N, "x");
    expect(real(N, 1.5) == 1.5, "unparsable real: default");
    put(N, "zips:1");
    expect(text(N) == "zips:1", "text as given");

    put(N, nullptr);
    std::printf(failures ? "%d FAILED\n" : "env.h: all rules hold\n", failures);
    return failures ? 1 : 0;
}
