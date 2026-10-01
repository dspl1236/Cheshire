# Cheshire: the "Fast Ransac" pipeline templates for Meshroom 2025.1 (0.3.9), written by meshroom-pair.cmd
# and removed by its --unpair.
#
#   meshroom-templates.ps1 <Meshroom dir> install|remove
#
# Meshroom 2025.1 raised two RANSAC limits to 50000: FeatureMatching's maxIteration (2048 in 2023.3) and
# StructureFromMotion's localizerEstimatorMaxIterations (4096). On the sets measured (docs/04, 0.3.6 and
# 0.3.9) the larger budgets verify more weak pairs and place no more views, for minutes of matching. The
# paired nodes keep Meshroom's defaults, so a paired run of Meshroom's own templates gives what a stock
# run gives. These templates are the opt-in instead: the installed Photogrammetry and Photogrammetry Draft
# templates, copied with the two 2023.3 values set in the graph, where they show and can be edited.
#
# Each is the stock file plus two lines, so a diff shows exactly what changed: "maxIteration": 2048 as
# the first input of the FeatureMatching node and "localizerEstimatorMaxIterations": 4096 as the first of
# the StructureFromMotion node, indented like their neighbours, with the stock file's line endings. They
# are built from the Meshroom being paired, so their node versions are that Meshroom's. Meshroom labels a
# template by splitting its file name at capitals: photogrammetryFastRansac.mg is "Photogrammetry Fast
# Ransac". scripts/linux/meshroom-pair.sh writes the same files, byte for byte apart from line endings.
#
# Meshroom 2023.3 keeps its templates in lib\meshroom\pipelines and already runs these counts: nothing to
# write there. A template whose graph is not the expected one (one node of each type, neither value set)
# is skipped with a message rather than guessed at.
param([Parameter(Mandatory)][string]$Meshroom, [ValidateSet('install', 'remove')][string]$Mode = 'install')
$ErrorActionPreference = 'Stop'

$dir = Join-Path $Meshroom 'aliceVision\share\meshroom'
$variants = @(
    @{ Stock = 'photogrammetry'; Name = 'photogrammetryFastRansac' },
    @{ Stock = 'photogrammetryDraft'; Name = 'photogrammetryDraftFastRansac' }
)
$sets = @(
    @{ NodeType = 'FeatureMatching'; Input = 'maxIteration'; Value = 2048 },
    @{ NodeType = 'StructureFromMotion'; Input = 'localizerEstimatorMaxIterations'; Value = 4096 }
)
$utf8 = New-Object System.Text.UTF8Encoding $false

# ours: one of our names, carrying both values (a file the user made under that name is left alone)
function Test-Ours([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return $false }
    $t = [System.IO.File]::ReadAllText($path)
    return $t.Contains('"maxIteration": 2048') -and $t.Contains('"localizerEstimatorMaxIterations": 4096')
}

if ($Mode -eq 'remove') {
    foreach ($v in $variants) {
        $dst = Join-Path $dir "$($v.Name).mg"
        if (Test-Ours $dst) { Remove-Item -LiteralPath $dst; "removed the pipeline template $($v.Name).mg" }
    }
    exit 0
}

if (-not (Test-Path -LiteralPath (Join-Path $dir 'photogrammetry.mg'))) {
    "no aliceVision\share\meshroom\photogrammetry.mg: not Meshroom 2025.1 (2023.3 already runs 2048/4096), no Fast Ransac template written"
    exit 0
}

# Insert `"<name>": <value>,` as the first entry of the inputs of the one node of that type; $null when
# the graph is not the expected shape. Ordinal searches: .NET Framework's default string search is
# culture-sensitive.
function Add-Input([string]$text, [string]$eol, [string]$nodeType, [string]$inputName, [int]$value) {
    $ord = [System.StringComparison]::Ordinal
    $tag = "`"nodeType`": `"$nodeType`","
    $i = $text.IndexOf($tag, $ord)
    if ($i -lt 0 -or $text.IndexOf($tag, $i + 1, $ord) -ge 0) { return $null }
    $open = "`"inputs`": {" + $eol
    $j = $text.IndexOf($open, $i, $ord)
    $next = $text.IndexOf('"nodeType":', $i + $tag.Length, $ord)
    if ($j -lt 0 -or ($next -ge 0 -and $next -lt $j)) { return $null }
    $lineStart = $text.LastIndexOf($eol, $j, $ord) + $eol.Length
    $indent = $text.Substring($lineStart, $j - $lineStart) + '    '
    return $text.Insert($j + $open.Length, "$indent`"$inputName`": $value," + $eol)
}

foreach ($v in $variants) {
    $src = Join-Path $dir "$($v.Stock).mg"
    $dst = Join-Path $dir "$($v.Name).mg"
    if (-not (Test-Path -LiteralPath $src)) { "no $($v.Stock).mg in $dir`: $($v.Name).mg not written"; continue }
    if ((Test-Path -LiteralPath $dst) -and -not (Test-Ours $dst)) { "$($v.Name).mg exists and is not Cheshire's: left alone"; continue }
    $text = [System.IO.File]::ReadAllText($src)
    $eol = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }
    $stock = $text | ConvertFrom-Json
    $new = $text
    foreach ($s in $sets) {
        $nodes = @($stock.graph.PSObject.Properties | Where-Object { $_.Value.nodeType -eq $s.NodeType })
        if ($nodes.Count -ne 1 -or $null -ne $nodes[0].Value.inputs.($s.Input)) { $new = $null; break }
        $new = Add-Input $new $eol $s.NodeType $s.Input $s.Value
        if ($null -eq $new) { break }
    }
    if ($null -eq $new) { "$($v.Stock).mg is not the graph this expects (one FeatureMatching and one StructureFromMotion, neither value set): $($v.Name).mg not written"; continue }
    # the result must parse, carry both values, and be the stock graph otherwise
    $g = $new | ConvertFrom-Json
    $ok = $true
    foreach ($s in $sets) {
        $n = @($g.graph.PSObject.Properties | Where-Object { $_.Value.nodeType -eq $s.NodeType })[0].Value
        if ($n.inputs.($s.Input) -ne $s.Value) { $ok = $false }
        $n.inputs.PSObject.Properties.Remove($s.Input)
    }
    if (-not $ok -or (($g | ConvertTo-Json -Depth 64 -Compress) -ne ($stock | ConvertTo-Json -Depth 64 -Compress))) {
        "$($v.Name).mg: the edit did not come out as the stock graph plus the two values: not written"; continue
    }
    [System.IO.File]::WriteAllText($dst, $new, $utf8)
    "installed the pipeline template `"$((Get-Culture).TextInfo.ToTitleCase(($v.Name -creplace '([A-Z])', ' $1')))`" ($($v.Name).mg): $($v.Stock).mg with FeatureMatching maxIteration 2048 and StructureFromMotion localizerEstimatorMaxIterations 4096, Meshroom 2023.3's RANSAC counts"
}
exit 0
