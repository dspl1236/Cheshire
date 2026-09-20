# Assemble a portable Windows CUDA package for bench-pc's GTX 1050 Ti.
#
# The build output dir already co-locates every AliceVision and vcpkg DLL. The only thing missing
# for a machine without the toolkit is cudart64_12.dll - dumpbin says aliceVision_depthMap_cuda.dll
# needs that and nothing else from CUDA (no cublas, no nvrtc), which keeps this to 0.6 MB extra
# rather than the 700+ MB cublasLt would have cost.
$ErrorActionPreference = "Stop"
$src  = "D:\MMI\cheshire\build\av-cuda\Windows-AMD64"
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
$stage = "D:\MMI\cheshire\build\pkg-cuda-windows"
$tarball = "D:\MMI\cheshire\build\cheshire-alicevision-cuda-windows-x64-cuda12.9.tar.gz"

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null

Write-Output "=== staging build output"
robocopy $src $stage /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }

Write-Output "=== adding the CUDA runtime"
Copy-Item "$cuda\cudart64_12.dll" $stage
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
    Copy-Item $popsift $stage -Force
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
$omp = Get-ChildItem "$vs\*\debug_nonredist\x64\Microsoft.VC*.OpenMP.LLVM\libomp140.x86_64.dll" | Sort-Object FullName | Select-Object -Last 1
if (-not $crt -or -not $omp) { throw "could not locate the MSVC runtime redistributables under $vs" }
Write-Output "=== adding the MSVC runtime"
Copy-Item "$($crt.FullName)\*.dll" $stage -Force
Copy-Item $omp.FullName $stage -Force
Write-Output "  from $($crt.Name) + libomp140.x86_64.dll"

# share/aliceVision carries the OCIO config and sensor database the binaries look for
$share = "D:\MMI\cheshire\build\av-cuda-install\share"
if (Test-Path $share) {
  robocopy $share "$stage\share" /E /NFL /NDL /NJH /NJS /NP | Out-Null
  Write-Output "=== added share/ (OCIO config, sensor database)"
}

$n = (Get-ChildItem $stage -Recurse -File).Count
$mb = [math]::Round((Get-ChildItem $stage -Recurse -File | Measure-Object Length -Sum).Sum/1MB,1)
Write-Output "=== staged: $n files, $mb MB"

Write-Output "=== compressing"
if (Test-Path $tarball) { Remove-Item $tarball -Force }
tar -czf $tarball -C (Split-Path $stage) (Split-Path $stage -Leaf)
"  $tarball  ($([math]::Round((Get-Item $tarball).Length/1MB,1)) MB)"
