# Cheshire: the "Fast Ransac" pipeline templates for Meshroom 2025.1 and later (0.3.9), written by
# meshroom-pair.cmd and removed by its --unpair.
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
# Upstream's new SfM pipeline replaces StructureFromMotion with SfMExpanding, whose resection takes the
# same localizerEstimatorMaxIterations, also 50000 by default. Meshroom 2025.1 ships it as Photogrammetry
# Experimental; upstream's nightly makes it Photogrammetry itself. A graph with SfMExpanding instead of
# StructureFromMotion gets the 4096 there, and Photogrammetry Experimental gets a Fast Ransac copy too
# (after 0.3.9, docs/04). A Meshroom without that template (2023.3, the nightly) is told so and goes on.
#
# Each is the stock file plus two lines, so a diff shows exactly what changed: "maxIteration": 2048 as
# the first input of the FeatureMatching node and "localizerEstimatorMaxIterations": 4096 as the first of
# the StructureFromMotion (or SfMExpanding) node, indented like their neighbours, with the stock file's
# line endings. They are built from the Meshroom being paired, so their node versions are that
# Meshroom's. Meshroom labels a template by splitting its file name at capitals:
# photogrammetryFastRansac.mg is "Photogrammetry Fast Ransac". scripts/linux/meshroom-pair.sh writes the
# same files, byte for byte apart from line endings.
#
# Meshroom 2023.3 keeps its templates in lib\meshroom\pipelines and already runs these counts: nothing to
# write there. A template whose graph is not the expected one (one FeatureMatching, one StructureFromMotion
# or SfMExpanding, neither value set) is skipped with a message rather than guessed at.
param([Parameter(Mandatory)][string]$Meshroom, [ValidateSet('install', 'remove')][string]$Mode = 'install')
$ErrorActionPreference = 'Stop'
# "C:\Meshroom\" remove reaches PowerShell as one argument, C:\Meshroom" remove, with $Mode left at install:
# meshroom-pair.cmd strips the trailing backslash, and anything else that slips through stops here.
if ($Meshroom.Contains('"')) { "the Meshroom folder arrived with a quote in it ($Meshroom), from a trailing backslash before the closing quote: no Fast Ransac template handled"; exit 1 }

$dir = Join-Path $Meshroom 'aliceVision\share\meshroom'
$variants = @(
    @{ Stock = 'photogrammetry'; Name = 'photogrammetryFastRansac' },
    @{ Stock = 'photogrammetryDraft'; Name = 'photogrammetryDraftFastRansac' },
    @{ Stock = 'photogrammetryExperimental'; Name = 'photogrammetryExperimentalFastRansac' }
)
# each value goes to the one node of its types (the SfM node is one or the other)
$sets = @(
    @{ NodeTypes = @('FeatureMatching'); Input = 'maxIteration'; Value = 2048 },
    @{ NodeTypes = @('StructureFromMotion', 'SfMExpanding'); Input = 'localizerEstimatorMaxIterations'; Value = 4096 }
)
$utf8 = New-Object System.Text.UTF8Encoding $false

# ours: one of our names, a file, carrying both values as whole numbers (a file the user made under that name is
# left alone; a substring test took "maxIteration": 20480 for ours). Matched on the bytes, one character per
# byte, as the Linux script matches them: ReadAllText would decode a UTF-16 file the Linux script cannot read.
function Test-Ours([string]$path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
    $t = [System.Text.Encoding]::GetEncoding(28591).GetString([System.IO.File]::ReadAllBytes($path))
    return [regex]::IsMatch($t, '"maxIteration": 2048(?![0-9.eE])') -and [regex]::IsMatch($t, '"localizerEstimatorMaxIterations": 4096(?![0-9.eE])')
}

# A JSON object's member by its exact name, as the Linux script's dict lookups take it ($o.name ignores case,
# reaches into arrays and accepts any type); $null when $o is not an object or has no such member. The comma
# keeps an array value whole.
$pso = [System.Management.Automation.PSCustomObject]
function Get-Exact($o, [string]$name) {
    if ($o -isnot $pso) { return $null }
    foreach ($p in $o.PSObject.Properties) { if ($p.Name -ceq $name) { return ,$p.Value } }
    return $null
}
# the nodes of a graph whose nodeType is one of $types, exactly: every match, so the caller can count them
function Get-Nodes($graph, $types) {
    foreach ($p in $graph.PSObject.Properties) {
        $t = Get-Exact $p.Value 'nodeType'
        if ($t -is [string] -and $types -ccontains $t) { ,$p.Value }
    }
}

if ($Mode -eq 'remove') {
    foreach ($v in $variants) {
        $dst = Join-Path $dir "$($v.Name).mg"
        if (Test-Ours $dst) { Remove-Item -LiteralPath $dst; "removed the pipeline template $($v.Name).mg" }
    }
    exit 0
}

if (-not (Test-Path -LiteralPath (Join-Path $dir 'photogrammetry.mg'))) {
    "no aliceVision\share\meshroom\photogrammetry.mg: not Meshroom 2025.1 or later (2023.3 already runs 2048/4096), no Fast Ransac template written"
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
    # ConvertFrom-Json returns nothing for an empty or blank file where Python's json.loads raises
    $stock = $null
    $parsed = -not [string]::IsNullOrWhiteSpace($text)
    if ($parsed) { try { $stock = $text | ConvertFrom-Json } catch { $parsed = $false } }
    if (-not $parsed) { "$($v.Stock).mg does not parse as JSON: $($v.Name).mg not written"; continue }
    $new = $text
    $why = $null  # 'shape': not the graph this expects; otherwise the node type whose text is not laid out as expected
    $types = @{}  # the node type each value went to
    $graph = Get-Exact $stock 'graph'
    if ($graph -isnot $pso) { $why = 'shape' }
    foreach ($s in $(if ($why) { @() } else { $sets })) {
        $nodes = @(Get-Nodes $graph $s.NodeTypes)
        # inputs absent or null: nothing set (the text insertion then finds no inputs and says so); anything but an
        # object, or the input already there under any case (PowerShell's parser refuses two keys differing only
        # in case): not the graph this expects
        $in = $null
        if ($nodes.Count -eq 1) { $in = Get-Exact $nodes[0] 'inputs' }
        if ($nodes.Count -ne 1 -or ($null -ne $in -and ($in -isnot $pso -or $in.PSObject.Properties.Name -contains $s.Input))) { $why = 'shape'; break }
        $types[$s.Input] = Get-Exact $nodes[0] 'nodeType'
        $new = Add-Input $new $eol $types[$s.Input] $s.Input $s.Value
        if ($null -eq $new) { $why = $types[$s.Input]; break }
    }
    if ($why -eq 'shape') { "$($v.Stock).mg is not the graph this expects (one FeatureMatching, one StructureFromMotion or SfMExpanding, neither value set): $($v.Name).mg not written"; continue }
    if ($why) { "$($v.Stock).mg: its $why node is not laid out as this expects (its `"nodeType`" line, then `"inputs`": { ending a line): $($v.Name).mg not written"; continue }
    # the result must parse, carry both values, and be the stock graph otherwise; anything that throws here
    # (an inputs object the insertion broke) is the same verdict, not the end of the script
    $same = $false
    try {
        $g = $new | ConvertFrom-Json
        $gg = Get-Exact $g 'graph'
        $ok = $true
        foreach ($s in $sets) {
            $n = @(Get-Nodes $gg @($types[$s.Input]))[0]
            $ni = Get-Exact $n 'inputs'
            if ((Get-Exact $ni $s.Input) -ne $s.Value) { $ok = $false }
            $ni.PSObject.Properties.Remove($s.Input)
        }
        $same = $ok -and (($g | ConvertTo-Json -Depth 64 -Compress) -ceq ($stock | ConvertTo-Json -Depth 64 -Compress))
    } catch { $same = $false }
    if (-not $same) {
        "$($v.Name).mg: the edit did not come out as the stock graph plus the two values: not written"; continue
    }
    [System.IO.File]::WriteAllText($dst, $new, $utf8)
    "installed the pipeline template `"$((Get-Culture).TextInfo.ToTitleCase(($v.Name -creplace '([A-Z])', ' $1')))`" ($($v.Name).mg): $($v.Stock).mg with $($types['maxIteration']) maxIteration 2048 and $($types['localizerEstimatorMaxIterations']) localizerEstimatorMaxIterations 4096, Meshroom 2023.3's RANSAC counts"
}
exit 0
