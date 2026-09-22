# Assemble a portable Windows CUDA package.
#
# The build output dir already co-locates every AliceVision and vcpkg DLL. The only thing missing
# for a machine without the toolkit is cudart64_12.dll - dumpbin says aliceVision_depthMap_cuda.dll
# needs that and nothing else from CUDA (no cublas, no nvrtc), which keeps this to 0.6 MB extra
# rather than the 700+ MB cublasLt would have cost.
$ErrorActionPreference = "Stop"
$src  = "D:\MMI\cheshire\build\av-cuda\Windows-AMD64"
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
# The stage directory name is what a user sees after extracting, so it is the release name and not
# a build-tree one: the zip used to unpack to a folder called pkg-cuda-windows.
$stage = "D:\MMI\cheshire\build\cheshire-alicevision-cuda-windows-x64"
$zip = "D:\MMI\cheshire\build\cheshire-alicevision-cuda-windows-x64-cuda12.9.zip"

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null

# bin\ + share\, the same shape as the HIP packages. This used to stage the build output flat at
# the package root, because it began as a one-off for bench-pc rather than a release artifact.
# Everything downstream assumes bin\: meshroom-pair.cmd probes <pkg>\bin\aliceVision_*.exe and the
# launcher runs <pkg>\bin\<node>.exe, so a flat package cannot be paired with Meshroom at all - it
# had to be faked with a directory junction to be tested end to end.
$bin = Join-Path $stage "bin"
New-Item -ItemType Directory -Path $bin | Out-Null

Write-Output "=== staging build output"
robocopy $src $bin /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }

Write-Output "=== adding the CUDA runtime"
Copy-Item "$cuda\cudart64_12.dll" $bin
"  cudart64_12.dll"

# PopSIFT, taken from its own install tree rather than from the build output.
#
# There is a prebuilt popsift.dll in the vcpkg dependency tree
# (tools/vcpkg-deps/.../bin/popsift.dll, 15.7 MB, no GPU runtime in its import table), and vcpkg's
# applocal deployment copies it into Windows-AMD64 on every AliceVision build - so the CUDA build
# linked against the CUDA PopSIFT we built and then *shipped* that one instead. On the GTX 1050 Ti
# that crashed with 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) on the first photograph, after
# printing "Choosing device 0". A CUDA build of PopSIFT is 3.6 MB and imports cudart64_12.dll;
# if what lands here is 15.7 MB, the wrong one has won again.
$popsift = "D:\MMI\cheshire\build\popsift-cuda-install\bin\popsift.dll"
if (Test-Path $popsift) {
    Copy-Item $popsift $bin -Force
    Write-Output ("=== PopSIFT from its install tree ({0:N2} MB)" -f ((Get-Item $popsift).Length/1MB))
} else {
    Write-Output "=== no CUDA PopSIFT install - GPU SIFT will fall back to the CPU extractor"
}

# MSVC runtime. The build machine has these in system32 from its VS install, which is why the
# package ran here and died on bench-pc with 0xC0000135 before a single log line. libomp140 is the
# one that actually bites: it is the LLVM OpenMP runtime CMake's FindOpenMP selects via
# -openmp:llvm (OpenMP 3.0+, which AliceVision needs - the redistributable vcomp140 is 2.0 only),
# and it ships with Visual Studio rather than the VC++ redistributable, so a machine with only the
# redist does not have it. The shipped HIP Windows packages already bundle the same set.
$vs = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Redist\MSVC"
$crt = Get-ChildItem "$vs\*\x64\Microsoft.VC*.CRT" -Directory | Sort-Object Name | Select-Object -Last 1
if (-not $crt) { throw "could not locate the MSVC CRT redistributable under $vs" }
Write-Output "=== adding the MSVC runtime"
Copy-Item "$($crt.FullName)\*.dll" $bin -Force
Write-Output "  from $($crt.Name)"

# OpenMP runtime: LLVM's own build, NOT Microsoft's. Microsoft's copy of the same runtime lives
# under a debug_nonredist path, which is their marker for files outside the Distributable Code
# terms - not ours to ship, and attribution would not change that. LLVM's is Apache-2.0 WITH
# LLVM-exception, redistributable on condition of shipping the licence, which is why LICENSE.TXT
# goes in beside it. See third_party/llvm-openmp/README.md.
#
# Copied under the name the binaries import (-openmp:llvm emits libomp140.x86_64.dll). Verified by
# comparing imports, not exports: 80 binaries import 32 symbols and LLVM's runtime provides all 32.
$ompSrc = "D:\MMI\cheshire\third_party\llvm-openmp"
if (-not (Test-Path "$ompSrc\libomp.dll")) {
    throw "no LLVM OpenMP runtime at $ompSrc - see its README for how to fetch one"
}
Copy-Item "$ompSrc\libomp.dll" (Join-Path $bin "libomp140.x86_64.dll") -Force
Copy-Item "$ompSrc\LICENSE.TXT" (Join-Path $stage "LICENSE.llvm-openmp.txt") -Force
Write-Output ("  LLVM OpenMP runtime {0:N2} MB + its licence" -f ((Get-Item "$ompSrc\libomp.dll").Length/1MB))

# Pairing: the script and the launcher, the same two files the HIP zip carries. The v0.3.0 CUDA
# Windows zip shipped WITHOUT them (492 entries, none for pairing) while the README sent NVIDIA
# users to meshroom-pair.cmd; and the copy that reached bench-pc by hand was a launcher built
# before CHESHIRE_BACKEND existed, which on an NVIDIA box hands every node back to Meshroom -
# a two-hour full-resolution run on the 1050 Ti tested nothing (2026-09-20). So the launcher is
# checked by substance: it must carry the CHESHIRE_BACKEND string (wide, as the exe stores it),
# not merely exist.
$launcher = "D:\MMI\cheshire\build\meshroom-pair-launcher.exe"
if (-not (Test-Path $launcher)) { throw "no launcher at $launcher - run scripts\windows\build-launcher.cmd" }
$bytes = [IO.File]::ReadAllBytes($launcher)
$wide = [Text.Encoding]::Unicode.GetString($bytes)
if ($wide -notmatch 'CHESHIRE_BACKEND') {
    throw "$launcher predates CHESHIRE_BACKEND (no such string in it) - rebuild it before packaging"
}
Copy-Item $launcher (Join-Path $stage "meshroom-pair-launcher.exe") -Force
Copy-Item "D:\MMI\cheshire\scripts\windows\meshroom-pair.cmd" (Join-Path $stage "meshroom-pair.cmd") -Force
Write-Output ("=== pairing: meshroom-pair.cmd + launcher ({0:N0} bytes, knows CHESHIRE_BACKEND)" -f $bytes.Length)

# GPU SIFT by substance: if a popsift.dll is staged, the feature library must import it. The v0.3.2
# zip staged popsift.dll (this script copies it from its install tree regardless) while
# aliceVision_feature.dll had been built with ALICEVISION_USE_POPSIFT=OFF and imported nothing -
# a package that says GPU SIFT and runs the CPU extractor. Refuse that here.
if (Test-Path (Join-Path $bin "popsift.dll")) {
    $feat = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes((Join-Path $bin "aliceVision_feature.dll")))
    if ($feat -notmatch 'popsift\.dll') {
        throw "aliceVision_feature.dll does not import popsift.dll: the build ran with ALICEVISION_USE_POPSIFT=OFF (set CHESHIRE_POPSIFT=ON and rebuild)"
    }
    Write-Output "=== GPU SIFT: aliceVision_feature.dll imports popsift.dll"
}

# GPL contamination guard (2026-09-22): the 0.3.0-0.3.2 Windows packages carried libspqr.dll and
# libcholmod.dll through vcpkg's Ceres. 0.3.3's Ceres is built without SuiteSparse; refuse a stage
# where a DLL still imports them (alicevision discussion #2116: no SPQR in pre-built binaries).
$gpl = Get-ChildItem $bin -Filter *.dll | Where-Object { $_.Name -in @("libspqr.dll", "libcholmod.dll", "libumfpack.dll", "libklu.dll") }
if ($gpl) {
    throw "GPL SuiteSparse libraries in the package: $($gpl.Name -join ', ') - rebuild Ceres without SuiteSparse first"
}
Write-Output "=== no SuiteSparse GPL libraries in the package"

# share/aliceVision carries the OCIO config and sensor database the binaries look for
$share = "D:\MMI\cheshire\build\av-cuda-install\share"
if (Test-Path $share) {
  robocopy $share "$stage\share" /E /NFL /NDL /NJH /NJS /NP | Out-Null
  Write-Output "=== added share/ (OCIO config, sensor database)"
}

$n = (Get-ChildItem $stage -Recurse -File).Count
$mb = [math]::Round((Get-ChildItem $stage -Recurse -File | Measure-Object Length -Sum).Sum/1MB,1)
Write-Output "=== staged: $n files, $mb MB"

# A zip, like the Windows HIP package: Explorer opens one without a tool, and a .tar.gz asset on a
# Windows release is a papercut even though tar.exe has shipped in Windows since 1809. bsdtar's -a
# picks the format from the extension, which is far faster than Compress-Archive at this size.
Write-Output "=== compressing"
if (Test-Path $zip) { Remove-Item $zip -Force }
# Windows' own tar (bsdtar) by full path: with Git's usrin ahead on PATH the name resolves to GNU tar,
# which reads 'D:' as a remote host and produces nothing (2026-09-22, the 0.3.3 package chain).
& (Join-Path $env:SystemRoot 'System32	ar.exe') -a -c -f $zip -C (Split-Path $stage) (Split-Path $stage -Leaf)
if (-not (Test-Path $zip)) { throw "tar produced no archive at $zip" }
"  $zip  ($([math]::Round((Get-Item $zip).Length/1MB,1)) MB)"
"  sha256: $((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower())"
