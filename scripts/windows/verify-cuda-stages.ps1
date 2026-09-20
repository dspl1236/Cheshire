# Run every GPU-bearing stage from a Windows CUDA package and check each one actually worked.
#
# Usage: verify-cuda-stages.ps1 [-Package <dir>] [-Cache <meshroom cache>] [-Photos <folder>]
#                               [-Views <n>] [-SkipHeavy]
#
# This exists because the GPU SIFT crash of 2026-09-21 would have been caught here and was caught
# nowhere else: the package linked a CUDA PopSIFT and shipped the 15.7 MB prebuilt from the vcpkg
# dependency tree, so FeatureExtraction died with 0xC0000409 on the first photograph while depth
# maps - the only thing being tested - were bit-identical across two platforms. A package can be
# flawless in six stages and broken in the seventh.
#
# Each stage must produce output AND emit a line proving the GPU path ran. Two rules, both learned
# the hard way, and both of which this script got wrong until the HIP gate was first run on
# 2026-09-20:
#
#   1. The marker must be a line the port emits UNCONDITIONALLY. 'filter votes GPU' looked right
#      but only appears under CHESHIRE_GPU_FILTER_DEBUG=1, so it reported a perfectly good GPU run
#      as a failure. Same for 'cheshire texturing profile' (CHESHIRE_GPU_TEX_LOG).
#   2. The marker must NOT also match the port's DISABLED message - the ports announce both, a few
#      words apart. 'GPU brute-force' matches "GPU brute-force disabled by CHESHIRE_GPU_MATCHER=0",
#      so a run that fell back to the CPU matcher passed as a GPU one. Likewise 'gpu|popsift'
#      matched any path containing 'gpu', and this machine has a C:\cheshire\gpusift.
#
# The rule that satisfies both: take the marker from the success branch of the port's available(),
# and include enough of it that the disabled branch cannot match.
#
# Counting files is not enough either: that passed a Texturing run which generated no textures at
# all, because --colorMappingFileType defaults to NONE.
#
# A reference cache usually only carries the nodes someone needed at the time, so stages bootstrap
# their inputs where they can - the feature stages from a photo folder, and the later depth stages
# from the output of the stage before them. Missing prerequisites are reported as skips, never as
# failures, and never as a crash, which is what the first version of this script did.
param(
    [string]$Package = "C:\cheshire\cuda-win",
    [string]$Cache   = "C:\cheshire\ref\enginebay-cache",
    [string]$Photos  = "C:\cheshire\data\monstree",
    [int]$Views      = 2,
    [switch]$SkipHeavy
)
$ErrorActionPreference = "Continue"

function Node($name) {
    $d = Get-ChildItem (Join-Path $Cache $name) -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($d) { return $d.FullName }
    return $null
}

$out = "C:\cheshire\stages-cuda-win"
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory $out | Out-Null
$env:ALICEVISION_ROOT = $Package
# Packages built before the bin\ layout staged everything flat at the root; accept either, so this
# keeps working against an unpacked older package.
$exeDir = if (Test-Path (Join-Path $Package "bin\aliceVision_meshing.exe")) { Join-Path $Package "bin" } else { $Package }

Write-Output "=== package: $Package"
Write-Output "=== cache:   $Cache"
$ps = Get-Item (Join-Path $exeDir "popsift.dll") -ErrorAction SilentlyContinue
if ($ps) {
    $mb = [math]::Round($ps.Length/1MB,2)
    $v = if ($mb -gt 10) { "*** 15.7 MB = the vcpkg prebuilt, NOT a CUDA build ***" } else { "ok (CUDA build)" }
    Write-Output ("  popsift.dll {0} MB  {1}" -f $mb, $v)
} else { Write-Output "  popsift.dll absent - GPU SIFT cannot work" }
& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | ForEach-Object { "  gpu: $_" }

$results = [ordered]@{}
function Stage($label, $exe, $glob, $expect, [string[]]$argv) {
    $dest = Join-Path $out $exe
    New-Item -ItemType Directory $dest -Force | Out-Null
    $log = Join-Path $out "$exe.log"
    $t0 = Get-Date
    & (Join-Path $exeDir "$exe.exe") @argv --verboseLevel info *> $log
    $rc = $LASTEXITCODE
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds,1)
    $made = (Get-ChildItem (Join-Path $dest $glob) -ErrorAction SilentlyContinue).Count
    $saw = if ($expect) { [bool](Select-String -Path $log -Pattern $expect -Quiet -ErrorAction SilentlyContinue) } else { $true }
    $ok = ($rc -eq 0) -and ($made -gt 0) -and $saw
    Write-Host ("  {0}  {1,-34} exit={2,-4} produced {3,-4} {4}s" -f $(if ($ok){"ok  "}else{"FAIL"}), $label, $rc, $made, $secs)
    if (-not $ok) {
        if (($rc -eq 0) -and ($made -gt 0) -and (-not $saw)) {
            Write-Host "        ran and produced output, but nothing matched /$expect/ - the GPU path is silent"
        }
        Get-Content $log -Tail 3 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ("        " + $_.Trim()) }
    }
    $script:results[$label] = $ok
}
function Skip($label, $why) {
    Write-Host ("  skip  {0,-34} {1}" -f $label, $why)
    $script:results[$label] = $null
}

$sfm = $null; $pds = $null; $dmf = $null; $dm = $null
if (Node "StructureFromMotion") { $sfm = Join-Path (Node "StructureFromMotion") "sfm.abc" }
$pds = Node "PrepareDenseScene"
$dm  = Node "DepthMap"
$dmf = Node "DepthMapFilter"

Write-Output ""
Write-Output "=== feature stages (inputs bootstrapped from photos) ==="
$w = Join-Path $out "_bootstrap"
New-Item -ItemType Directory "$w\photos" -Force | Out-Null
$src = Get-ChildItem $Photos -Recurse -Include *.jpg,*.JPG,*.png -ErrorAction SilentlyContinue | Select-Object -First 5
if ($src.Count -lt 2) {
    Skip "FeatureExtraction (GPU SIFT)" "no photos under $Photos"
    Skip "FeatureMatching (GPU matcher)" "no photos under $Photos"
} else {
    $src | ForEach-Object { Copy-Item $_.FullName "$w\photos\" -Force }
    & (Join-Path $exeDir "aliceVision_cameraInit.exe") --imageFolder "$w\photos" `
        --sensorDatabase (Join-Path $Package "share\aliceVision\cameraSensors.db") `
        --defaultFieldOfView 45 --allowSingleView 1 --output "$w\cameraInit.sfm" --verboseLevel error *> "$w\cameraInit.log"
    if ($LASTEXITCODE -ne 0) {
        Skip "FeatureExtraction (GPU SIFT)" "cameraInit failed"
        Skip "FeatureMatching (GPU matcher)" "cameraInit failed"
    } else {
        $feOut = Join-Path $out "aliceVision_featureExtraction"
        # PopSIFT's device_prop_t::set(0, true) prints this whenever the describer is constructed,
        # so it is present exactly when GPU SIFT is in use and absent when it is not.
        Stage "FeatureExtraction (GPU SIFT)" "aliceVision_featureExtraction" "*.feat" "Choosing device \d+:" @(
            "--input","$w\cameraInit.sfm","--describerTypes","sift","--describerPreset","normal",
            "--forceCpuExtraction","False","--output",$feOut)
        $feOk = $results["FeatureExtraction (GPU SIFT)"]

        if (-not $feOk) { Skip "FeatureMatching (GPU matcher)" "needs FeatureExtraction output" }
        else {
            & (Join-Path $exeDir "aliceVision_imageMatching.exe") --input "$w\cameraInit.sfm" `
                --featuresFolders $feOut --method Exhaustive --output "$w\imageMatches.txt" --verboseLevel error *> "$w\imageMatching.log"
            if ($LASTEXITCODE -ne 0) { Skip "FeatureMatching (GPU matcher)" "imageMatching failed" }
            else {
                Stage "FeatureMatching (GPU matcher)" "aliceVision_featureMatching" "*.txt" "GPU brute-force L2 2-NN on" @(
                    "--input","$w\cameraInit.sfm","--featuresFolders",$feOut,"--imagePairsList","$w\imageMatches.txt",
                    "--describerTypes","sift","--geometricFilterType","fundamental_matrix",
                    "--output",(Join-Path $out "aliceVision_featureMatching"))
            }
        }
    }
}

Write-Output ""
Write-Output "=== depth stages (from the reference cache) ==="
$dmOut  = Join-Path $out "aliceVision_depthMapEstimation"
$dmfOut = Join-Path $out "aliceVision_depthMapFiltering"
if (-not $sfm -or -not (Test-Path $sfm) -or -not $pds) {
    Skip "DepthMap" "cache has no StructureFromMotion/PrepareDenseScene"
    Skip "DepthMapFilter (GPU filter)" "cache has no StructureFromMotion"
} else {
    Stage "DepthMap" "aliceVision_depthMapEstimation" "*.exr" "Number of GPU devices" @(
        "--input",$sfm,"--imagesFolder",$pds,"--downscale","2","--sgmDepthListPerTile","True",
        "--rangeStart","0","--rangeSize","$Views","--output",$dmOut)

    # Filtering consumes the RAW maps. This used to be handed the DepthMapFilter folder, so it
    # filtered already-filtered maps: it ran, produced output and reported ok while never seeing
    # the input the stage gets in a pipeline. Prefer the cache's DepthMap folder; failing that,
    # chain from the maps the stage above just made, which is the real pipeline edge anyway.
    $filterSrc = $null
    if ($dm) { $filterSrc = $dm }
    elseif ($results["DepthMap"]) { $filterSrc = $dmOut }
    if (-not $filterSrc) { Skip "DepthMapFilter (GPU filter)" "no raw depth maps: cache has no DepthMap and the DepthMap stage did not run" }
    else {
        Stage "DepthMapFilter (GPU filter)" "aliceVision_depthMapFiltering" "*.exr" "depth map filter: group votes on" @(
            "--input",$sfm,"--depthMapsFolder",$filterSrc,"--rangeStart","0","--rangeSize","$Views",
            "--output",$dmfOut)
    }
}

Write-Output ""
Write-Output "=== mesh stages (no range option - these are full runs) ==="
# Meshing and Texturing have no range option, so they process the whole scene however small the
# range above was. Leaving them out is exactly how GPU SIFT shipped broken, so they are here; on a
# large cache against a small card they are also the slow part, hence -SkipHeavy.
$meshDir = Join-Path $out "aliceVision_meshing"
$meshSrc = $null
if ($dmf) { $meshSrc = $dmf } elseif ($results["DepthMapFilter (GPU filter)"]) { $meshSrc = $dmfOut }
if ($SkipHeavy) {
    Skip "Meshing"  "-SkipHeavy"
    Skip "Texturing (GPU)" "-SkipHeavy"
} elseif (-not $sfm -or -not (Test-Path $sfm) -or -not $meshSrc) {
    Skip "Meshing" "no filtered depth maps"
    Skip "Texturing (GPU)" "needs Meshing output"
} else {
    Stage "Meshing" "aliceVision_meshing" "*.abc" "meshing votes: ray marching on" @(
        "--input",$sfm,"--depthMapsFolder",$meshSrc,
        "--outputMesh",(Join-Path $meshDir "mesh.obj"),
        "--maxPoints","1000000","--maxInputPoints","10000000","--estimateSpaceFromSfM","True",
        "--output",(Join-Path $meshDir "densePointCloud.abc"))

    # Texturing needs Meshing's densePointCloud.abc (not the SfM) for visibility, and
    # --colorMappingFileType or it writes texturedMesh.obj/.mtl, exits 0 and generates no textures.
    if (-not $results["Meshing"]) { Skip "Texturing (GPU)" "needs Meshing output" }
    else {
        Stage "Texturing (GPU)" "aliceVision_texturing" "texture_*.png" "texturing: pyramid \+ rasterisation on" @(
            "--input",(Join-Path $meshDir "densePointCloud.abc"),
            "--inputMesh",(Join-Path $meshDir "mesh.obj"),
            "--imagesFolder",$pds,"--colorMappingFileType","png",
            "--output",(Join-Path $out "aliceVision_texturing"))
    }
}

Write-Output ""
$ran     = ($results.Values | Where-Object { $_ -ne $null }).Count
$passed  = ($results.Values | Where-Object { $_ -eq $true }).Count
$skipped = ($results.Values | Where-Object { $_ -eq $null }).Count
Write-Output ("{0} of {1} stages ran from this package ({2} skipped for missing inputs)" -f $passed, $ran, $skipped)
if ($passed -lt $ran) { exit 1 }
