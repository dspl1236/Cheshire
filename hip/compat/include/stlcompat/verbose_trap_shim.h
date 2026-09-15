// Cheshire: MSVC 14.50's STL (Visual Studio 2026) calls __builtin_verbose_trap in its hardening
// checks when compiled by clang; clang 19 (HIP SDK 6.2 for Windows) does not have that builtin
// (it arrived in clang 20). For plain C++ TUs a force-include (/FI) of this header suffices.
// For HIP TUs it does not: clang pre-includes its HIP runtime wrapper before any /FI, and that
// wrapper already pulls the STL in, so the macro must come from the command line instead:
//   -D__builtin_verbose_trap(x,y)=__builtin_trap()
// (CHESHIRE_EXTRA_CXXFLAGS / CHESHIRE_HIP_EXTRA_FLAGS in build-alicevision.cmd). The trap path
// stays a trap either way.
#pragma once
#if defined(__clang__)
#  if !defined(__has_builtin) || !__has_builtin(__builtin_verbose_trap)
#    define __builtin_verbose_trap(category, reason) __builtin_trap()
#  endif
#endif
