# Cheshire: can this package's binary stand in for Meshroom's own? (called by meshroom-pair.cmd)
#
#   meshroom-pair-check.ps1 <Meshroom dir> <Cheshire package dir> <node> [option the launcher drops, without its dashes]
#
# A Meshroom node passes its binary the options its node description knows, and those are options
# Meshroom's own binary takes. So the package's binary must take every option Meshroom's does,
# except one the launcher strips (DepthMap's --sgmFilteringAxes). A newer Meshroom whose nodes pass
# options this package does not know would otherwise fail mid-job with "unrecognised option", and
# a node whose options kept their names but changed meaning is past what this can see. The check
# reads both binaries' --help, which needs no GPU. Exit 0: compatible, or it could not tell (no
# options from Meshroom's binary; pairing then goes ahead as before). Exit 1: options missing,
# printed.
param([Parameter(Mandatory = $true)][string]$Meshroom,
      [Parameter(Mandatory = $true)][string]$Package,
      [Parameter(Mandatory = $true)][string]$Node,
      [string]$Drop = "")

function Get-Options([string]$text) {
    [regex]::Matches($text, ' --[A-Za-z][A-Za-z0-9_.]*') | ForEach-Object { $_.Value.Trim() } | Sort-Object -Unique
}

$bin = Join-Path $Meshroom "aliceVision\bin"
# once paired, Meshroom's own binary is kept as <node>.cuda.exe and <node>.exe is the launcher
$own = Join-Path $bin "$Node.cuda.exe"
if (-not (Test-Path $own)) { $own = Join-Path $bin "$Node.exe" }
if (-not (Test-Path $own)) { Write-Output "${Node}: no binary in Meshroom's aliceVision\bin, compatibility not checked"; exit 0 }

$saved = @{ PATH = $env:PATH; ALICEVISION_ROOT = $env:ALICEVISION_ROOT }
try {
    $env:ALICEVISION_ROOT = Join-Path $Meshroom "aliceVision"
    $meshroomOptions = @(Get-Options ((& $own --help 2>&1) | Out-String))
    $env:PATH = $saved.PATH
    if (Test-Path (Join-Path $Package "cheshire-run.cmd")) {
        # a bundle: its binaries load only once the runner has composed PATH from the chosen payload
        $env:ALICEVISION_ROOT = $saved.ALICEVISION_ROOT
        $ours = (& cmd /c "`"$(Join-Path $Package 'cheshire-run.cmd')`" $Node --help" 2>&1) | Out-String
    } else {
        $env:PATH = (Join-Path $Package "bin") + ";" + $saved.PATH
        $env:ALICEVISION_ROOT = $Package
        $ours = (& (Join-Path $Package "bin\$Node.exe") --help 2>&1) | Out-String
    }
    $cheshireOptions = @(Get-Options $ours)
} finally {
    $env:PATH = $saved.PATH
    $env:ALICEVISION_ROOT = $saved.ALICEVISION_ROOT
}

if ($meshroomOptions.Count -eq 0) { Write-Output "${Node}: Meshroom's binary listed no options, compatibility not checked"; exit 0 }
if ($cheshireOptions.Count -eq 0) { Write-Output "${Node}: the package's binary listed no options (it did not start?)"; exit 1 }
# the dropped option comes without its dashes: PowerShell would read "--name" as a parameter
$dropped = if ($Drop) { "--" + $Drop.TrimStart('-') } else { "" }
$missing = @($meshroomOptions | Where-Object { $_ -notin $cheshireOptions -and $_ -ne $dropped })
if ($missing.Count -gt 0) {
    Write-Output ("${Node}: Meshroom's binary takes options this package's does not: " + ($missing -join " ") + " (a newer Meshroom than the package was built for?)")
    exit 1
}
exit 0
