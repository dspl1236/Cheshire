# Upstream draft: `bundle` target drops all but the first library search path

**Target:** alicevision/AliceVision, `CMakeLists.txt` (top level)
**Status:** draft, not submitted

## The change

```diff
 add_custom_target(bundle
-    ${CMAKE_COMMAND}
+    COMMAND ${CMAKE_COMMAND}
     -DBUNDLE_INSTALL_PREFIX=${ALICEVISION_BUNDLE_PREFIX}
     -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
-    -DBUNDLE_LIBS_PATHS=${BUNDLE_LIBS_PATHS}
+    "-DBUNDLE_LIBS_PATHS=${BUNDLE_LIBS_PATHS}"
     -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
     -P ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/MakeBundle.cmake
+    VERBATIM
 )
```

## Proposed PR title

`fix(cmake): bundle target passes only the first library search path`

## Proposed PR body

`BUNDLE_LIBS_PATHS` is a CMake list, and the `bundle` target passes it unquoted:

```cmake
-DBUNDLE_LIBS_PATHS=${BUNDLE_LIBS_PATHS}
```

An unquoted list expands to one argument per element, so `MakeBundle.cmake` receives only the
first path as the value of `BUNDLE_LIBS_PATHS`; every remaining element becomes a separate
argument, which `cmake -P` ignores. `LIBS_LOOKUPS_PATHS` is then short by however many paths were
configured, and `fixup_bundle` resolves dependencies against an incomplete search list.

This is silent. A dependency that happens to be resolvable another way - already beside the
executable, or reachable through the loader - is found anyway, so the bundle looks correct. It
only surfaces when a dependency lives *only* in one of the dropped paths, and then it surfaces as

```
warning: cannot resolve item 'libfoo.so.1.2.3'
CMake Error at BundleUtilities.cmake:740 (file):
  file READ_ELF given FILE "libfoo.so.1.2.3" that does not exist.
```

which does not obviously point at argument quoting.

Quoting alone is not sufficient. With a quoted argument the generator emits a single argument
containing `;` separators, and the build tool runs that command through a shell where `;`
separates commands - so the shell tries to execute the paths:

```
/bin/sh: 1: /opt/deps/lib64: Permission denied
```

The `COMMAND` form with `VERBATIM` is what makes CMake escape each argument correctly for the
native tool, on both POSIX shells and Windows.

### How it was found

Building AliceVision with `ALICEVISION_USE_POPSIFT=ON` against a PopSIFT installed outside the
dependency prefix. The extra prefix was configured through
`ALICEVISION_BUNDLE_SEARCH_LIBS_PATHS`, appeared correctly in `BUNDLE_LIBS_PATHS`, and was
dropped on the way to `MakeBundle.cmake`.

Verified before and after by reading the generated command out of `build.ninja`: before the
change the value ends at the first path and the rest are loose arguments; after it, the full
`a;b;c` list arrives as one argument and `fixup_bundle` completes.

### Scope

Any configuration with more than one entry in `BUNDLE_LIBS_PATHS` - which includes the common case
of a vcpkg bin directory plus a CUDA toolkit directory, where the toolkit path has been silently
absent from the search list.
