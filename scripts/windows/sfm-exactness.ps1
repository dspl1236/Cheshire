# Cheshire's own bundle-adjustment solver on a Windows test box: the exactness checks (0.3.7), the
# counterpart of scripts/linux/sfm-exactness.sh.
#
#   sfm-exactness.ps1 -Bin <package bin> -Root <root> [-Sets 41,eb]
#
# <Root>\<set>\cache holds CameraInit\cameraInit.sfm, FeatureExtraction and FeatureMatching (scripts/sfmbench.py's
# caches, copied over). Colours are not computed (the images stay behind), so the digests are this machine's own:
# the checks compare runs here with each other.
#   host      the host solver alone (CHESHIRE_BA_DEVICE=0)
#   host2     the same again (the run repeats itself)
#   device    the device on every solve it takes, from 1000 rows
#   devcheck  every such solve on the host from a snapshot, then on the device, compared
#   direct    the direct build against the Problem build, every solve
#   default   the defaults (the device from 200,000 rows), for the time
param([Parameter(Mandatory)][string]$Bin, [Parameter(Mandatory)][string]$Root, [string[]]$Sets = @('41', 'eb'))
# -File passes "41,eb" as one string
$Sets = @($Sets | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$env:PATH = "$Bin;$env:PATH"
$env:ALICEVISION_ROOT = Split-Path $Bin -Parent

function Run-Sfm([string]$Set, [string]$Tag, [hashtable]$Vars) {
    $C = Join-Path $Root "$Set\cache"
    $O = Join-Path $Root "$Set\runs\$Tag"
    if (Test-Path $O) { Remove-Item -Recurse -Force $O }
    New-Item -ItemType Directory -Force $O | Out-Null
    $saved = @{}
    $all = @{ 'CHESHIRE_BA_PROFILE' = '1' } + $Vars
    foreach ($k in $all.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $all[$k]) }
    $cmdArgs = @('--input', "$C\CameraInit\cameraInit.sfm", '--featuresFolders', "$C\FeatureExtraction", '--matchesFolders', "$C\FeatureMatching",
        '--describerTypes', 'dspsift', '--localizerEstimator', 'acransac', '--observationConstraint', 'Scale',
        '--localizerEstimatorMaxIterations', '4096', '--localizerEstimatorError', '0.0', '--lockScenePreviouslyReconstructed', 'False',
        '--useLocalBA', 'True', '--localBAGraphDistance', '1', '--nbFirstUnstableCameras', '30', '--maxImagesPerGroup', '30',
        '--bundleAdjustmentMaxOutliers', '50', '--maxNumberOfMatches', '0', '--minNumberOfMatches', '0', '--minInputTrackLength', '2',
        '--minNumberOfObservationsForTriangulation', '2', '--minAngleForTriangulation', '3.0', '--minAngleForLandmark', '2.0',
        '--maxReprojectionError', '4.0', '--minAngleInitialPair', '5.0', '--maxAngleInitialPair', '40.0',
        '--useOnlyMatchesFromInputFolder', 'False', '--useRigConstraint', 'True', '--rigMinNbCamerasForCalibration', '20',
        '--lockAllIntrinsics', 'False', '--minNbCamerasToRefinePrincipalPoint', '3', '--filterTrackForks', 'False',
        '--computeStructureColor', 'False', '--useAutoTransform', 'True',
        '--interFileExtension', '.abc', '--logIntermediateSteps', 'False', '--verboseLevel', 'info',
        '--output', "$O\sfm.abc", '--outputViewsAndPoses', "$O\cameras.sfm", '--extraInfoFolder', $O)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    # every argument is free of spaces, so a joined string is the command line
    $p = Start-Process -FilePath (Join-Path $Bin 'aliceVision_incrementalSfM.exe') -ArgumentList ($cmdArgs -join ' ') -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$O\stdout.log" -RedirectStandardError "$O\stderr.log"
    $rc = $p.ExitCode
    Get-Content "$O\stdout.log", "$O\stderr.log" | Set-Content "$O\sfm.log"
    $sw.Stop()
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    $abc = '-'
    if (Test-Path "$O\sfm.abc") { $abc = (Get-FileHash "$O\sfm.abc" -Algorithm SHA256).Hash.Substring(0, 16).ToLower() }
    $log = Get-Content "$O\sfm.log"
    $dev = @($log | Select-String 'BA device check')
    $devBad = @($dev | Where-Object { $_.Line -notmatch ': same' })
    $dir = @($log | Select-String 'BA direct check')
    $dirBad = @($dir | Where-Object { $_.Line -notmatch ': same' })
    $devOn = @($log | Select-String 'BA own profile' | Where-Object { $_.Line -match ', device' })
    '{0,-4} {1,-9} rc {2} wall {3,7:N1} s  abc {4}  device solves {5}  device check {6} (not same {7})  direct check {8} (not same {9})' -f `
        $Set, $Tag, $rc, $sw.Elapsed.TotalSeconds, $abc, $devOn.Count, $dev.Count, $devBad.Count, $dir.Count, $dirBad.Count
    $log | Select-String 'BA device:|device failed|OpenMP threads wait' | Select-Object -First 3 | ForEach-Object { '     ' + $_.Line }
}

foreach ($s in $Sets) {
    Run-Sfm $s 'host' @{ 'CHESHIRE_BA_DEVICE' = '0' }
    Run-Sfm $s 'host2' @{ 'CHESHIRE_BA_DEVICE' = '0' }
    Run-Sfm $s 'device' @{ 'CHESHIRE_BA_DEVICE_MIN_ROWS' = '1000' }
    Run-Sfm $s 'devcheck' @{ 'CHESHIRE_BA_DEVICE_MIN_ROWS' = '1000'; 'CHESHIRE_BA_DEVICE' = 'check' }
    Run-Sfm $s 'direct' @{ 'CHESHIRE_BA_DIRECT' = 'check' }
    Run-Sfm $s 'default' @{}
}
