# Per-process resource + render-time benchmark for the `window` example.
#
# Builds each requested revision in its own git worktree, runs the release
# window example for a fixed wall time, and samples the process's CPU, RAM,
# GPU and VRAM while it renders. Render time comes from the example's own
# "[heartbeat] frame N fps F" line, so it measures the render loop, not the
# sampler.
#
#   pwsh -File scripts/bench.ps1
#   pwsh -File scripts/bench.ps1 -Revs HEAD,011a0b86 -Seconds 60 -Serialize both
#
# Both revisions are normalized before building (same scene, same window size,
# heartbeat print injected where it is missing) so the only difference left is
# the renderer itself.

[CmdletBinding()]
param(
    # Revisions to compare. Anything `git rev-parse` accepts.
    [string[]]$Revs = @('HEAD', '011a0b86'),
    # Measured wall time per run, after the warmup.
    [int]$Seconds = 60,
    # Discarded lead-in: shader/pipeline warmup and the first frames are not steady state.
    [int]$Warmup = 15,
    [int]$Width = 1920,
    [int]$Height = 1080,
    # '1' = whole-frame serialization, '0' = overlapped frames, 'both' = one run each.
    # Only revisions that read SUNRAY_SERIALIZE_FRAMES are affected; the others
    # are labelled n/a and run once.
    [ValidateSet('0', '1', 'both')][string]$Serialize = '1',
    [string]$AliasStrategy = 'off',
    [ValidateSet('0', '1')][string]$Validation = '0',
    [int]$IntervalMs = 500,
    # Localized Windows installs name these differently; override if sampling reports n/a.
    [string]$GpuCounter = 'Utilization Percentage',
    [string]$GpuEngineSet = 'GPU Engine',
    [string]$VramCounter = 'Dedicated Usage',
    [string]$VramSet = 'GPU Process Memory',
    [string]$WorkDir = (Join-Path $env:TEMP 'sunray-bench')
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$repo = $PWD.Path
$cores = [Environment]::ProcessorCount

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# --- worktree + normalization ----------------------------------------------

# Pins scene, resolution and stdout heartbeat across revisions. Without this the
# old example renders a different (and now missing) .glb at a different size and
# only reports fps in the window title, where nothing can read it.
function Initialize-Worktree($rev) {
    $sha = (git rev-parse --short $rev).Trim()
    $dir = Join-Path $WorkDir $sha
    if (-not (Test-Path $dir)) {
        Step "Checking out $rev ($sha)"
        git worktree add --detach $dir $sha | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "git worktree add failed for $rev" }
    }

    $main = Join-Path $dir 'examples/window/main.rs'
    $src = Get-Content $main -Raw

    $src = $src -replace 'load_gltf\("[^"]*"\)', 'load_gltf("examples/assets/Room.glb")'
    $src = $src -replace 'LogicalSize::new\([^)]*\)', "LogicalSize::new($Width, $Height)"
    if ($src -notmatch '\[heartbeat\]') {
        $src = $src -replace '(?m)^(\s*)self\.last_fps_check = Some\(now\);', "`$1println!(`"[heartbeat] frame {} fps {fps:.1}`", self.frame_count);`r`n`$1self.last_fps_check = Some(now);"
    }
    Set-Content -Path $main -Value $src -Encoding utf8 -NoNewline

    if ($src -notmatch '\[heartbeat\]') { throw "$sha : could not inject the heartbeat print, nothing to measure" }

    Step "Building $sha (release)"
    Push-Location $dir
    try {
        cargo build --release --example window | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "build failed for $sha" }
    } finally { Pop-Location }

    [pscustomobject]@{
        Rev            = $rev
        Sha            = $sha
        Dir            = $dir
        Exe            = Join-Path $dir 'target/release/examples/window.exe'
        ReadsSerialize = $null -ne (Get-ChildItem -Path (Join-Path $dir 'src') -Filter *.rs -Recurse |
            Select-String -Pattern 'SUNRAY_SERIALIZE_FRAMES' -SimpleMatch -List | Select-Object -First 1)
    }
}

# --- sampling ---------------------------------------------------------------

# Per-process GPU load and dedicated VRAM. Both countersets are instanced per
# pid, so the wildcard gives exactly this process's engines; the engines are
# summed because a frame spans several (3D, compute, copy).
function Get-GpuSample($procId) {
    $gpu = $null; $vram = $null
    try {
        $s = (Get-Counter -Counter "\$GpuEngineSet(pid_${procId}*)\$GpuCounter" -ErrorAction Stop).CounterSamples
        $gpu = ($s | Measure-Object CookedValue -Sum).Sum
    } catch {}
    try {
        $s = (Get-Counter -Counter "\$VramSet(pid_${procId}*)\$VramCounter" -ErrorAction Stop).CounterSamples
        $vram = ($s | Measure-Object CookedValue -Sum).Sum
    } catch {}
    , @($gpu, $vram)
}

# ExitCode is only populated after a full WaitForExit; a driver crash shows up
# here as 0xC0000005 rather than a clean code.
function Format-Exit($p) {
    $p.WaitForExit()
    "(exit 0x{0:X8})" -f $p.ExitCode
}

function Invoke-Run($build, $serialize, $label) {
    $out = Join-Path $WorkDir "$($build.Sha)-$serialize.out"
    $err = Join-Path $WorkDir "$($build.Sha)-$serialize.err"

    $envVars = @{
        SUNRAY_ENABLE_VALIDATION_LAYER = $Validation
        SUNRAY_ENABLE_GPUAV            = '0'
        SUNRAY_ENABLE_NVIDIA_AFTERMATH = '0'
        SUNRAY_ENABLE_NSIGHT           = '0'
        SUNRAY_GRAPH_DUMP_DIR          = '0'
        SUNRAY_SERIALIZE_FRAMES        = $serialize
        SUNRAY_ALIAS_STRATEGY          = $AliasStrategy
    }
    foreach ($k in $envVars.Keys) { Set-Item "env:$k" $envVars[$k] }

    Step "Run: $label"
    $p = Start-Process -FilePath $build.Exe -WorkingDirectory $build.Dir -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err

    # Warmup is wall time, not samples: shader compilation and the first
    # swapchain frames are not what we are measuring.
    $deadline = [datetime]::UtcNow.AddSeconds($Warmup)
    while ([datetime]::UtcNow -lt $deadline -and -not $p.HasExited) { Start-Sleep -Milliseconds 200 }
    if ($p.HasExited) { throw "$label exited during warmup $(Format-Exit $p) -- see $err" }

    $cpuS = @(); $ramS = @(); $gpuS = @(); $vramS = @()
    $p.Refresh(); $prevCpu = $p.TotalProcessorTime; $prevT = [datetime]::UtcNow
    $stop = [datetime]::UtcNow.AddSeconds($Seconds)

    while ([datetime]::UtcNow -lt $stop) {
        Start-Sleep -Milliseconds $IntervalMs
        if ($p.HasExited) { throw "$label exited mid-run $(Format-Exit $p) -- see $err" }
        $p.Refresh()
        $now = [datetime]::UtcNow
        $cpuS += ($p.TotalProcessorTime - $prevCpu).TotalSeconds / ($now - $prevT).TotalSeconds / $cores * 100
        $prevCpu = $p.TotalProcessorTime; $prevT = $now
        $ramS += $p.WorkingSet64 / 1MB
        $g = Get-GpuSample $p.Id
        if ($null -ne $g[0]) { $gpuS += $g[0] }
        if ($null -ne $g[1]) { $vramS += $g[1] / 1MB }
    }

    $p.Kill(); $p.WaitForExit()

    # Frame time from the app's own counters: total frames and total time over
    # the post-warmup heartbeats, so long and short intervals weigh correctly.
    # One heartbeat per second, so skipping $Warmup lines drops the warmup.
    $hb = @(Select-String -Path $out -Pattern '\[heartbeat\] frame (\d+) fps ([\d.]+)' |
        Select-Object -Skip $Warmup |
        ForEach-Object { [pscustomobject]@{ Frame = [long]$_.Matches[0].Groups[1].Value; Fps = [double]$_.Matches[0].Groups[2].Value } })
    if ($hb.Count -lt 2) { throw "$label produced $($hb.Count) heartbeat(s) after warmup -- alive but not rendering?" }

    $frames = 0; $time = 0.0
    for ($i = 1; $i -lt $hb.Count; $i++) {
        $d = $hb[$i].Frame - $hb[$i - 1].Frame
        if ($d -le 0 -or $hb[$i].Fps -le 0) { continue }
        $frames += $d; $time += $d / $hb[$i].Fps
    }

    $avg = { param($a) if ($a.Count) { ($a | Measure-Object -Average).Average } else { $null } }
    $max = { param($a) if ($a.Count) { ($a | Measure-Object -Maximum).Maximum } else { $null } }

    [pscustomobject]@{
        Run       = $label
        FrameMs   = if ($frames) { 1000 * $time / $frames } else { $null }
        Fps       = if ($time) { $frames / $time } else { $null }
        Frames    = $frames
        CpuPct    = & $avg $cpuS
        CpuMaxPct = & $max $cpuS
        RamMB     = & $avg $ramS
        RamMaxMB  = & $max $ramS
        GpuPct    = & $avg $gpuS
        VramMB    = & $avg $vramS
        VramMaxMB = & $max $vramS
        Note      = ''
    }
}

# --- main -------------------------------------------------------------------

New-Item -ItemType Directory -Force $WorkDir | Out-Null
$builds = $Revs | ForEach-Object { Initialize-Worktree $_ }

$results = foreach ($b in $builds) {
    $modes = if ($Serialize -eq 'both' -and $b.ReadsSerialize) { @('1', '0') } else { @($Serialize) }
    foreach ($m in $modes) {
        $tag = if ($b.ReadsSerialize) { "serialize=$m" } else { 'serialize=n/a' }
        $label = "$($b.Sha) $tag"
        # A configuration that crashes is a result, not a reason to lose the
        # runs that did work -- record it and keep going.
        try { Invoke-Run $b $m $label }
        catch {
            Write-Host $_.Exception.Message -ForegroundColor Yellow
            [pscustomobject]@{ Run = $label; FrameMs = $null; Fps = $null; Frames = 0; CpuPct = $null
                CpuMaxPct = $null; RamMB = $null; RamMaxMB = $null; GpuPct = $null; VramMB = $null
                VramMaxMB = $null; Note = $_.Exception.Message
            }
        }
    }
}

Step 'Results'
$results | Format-Table -AutoSize @(
    'Run'
    @{ n = 'render ms'; e = { '{0:N2}' -f $_.FrameMs } }
    @{ n = 'fps'; e = { '{0:N1}' -f $_.Fps } }
    @{ n = 'cpu %'; e = { '{0:N1}' -f $_.CpuPct } }
    @{ n = 'cpu max %'; e = { '{0:N1}' -f $_.CpuMaxPct } }
    @{ n = 'ram MB'; e = { '{0:N0}' -f $_.RamMB } }
    @{ n = 'ram max MB'; e = { '{0:N0}' -f $_.RamMaxMB } }
    @{ n = 'gpu %'; e = { if ($null -eq $_.GpuPct) { 'n/a' } else { '{0:N1}' -f $_.GpuPct } } }
    @{ n = 'vram MB'; e = { if ($null -eq $_.VramMB) { 'n/a' } else { '{0:N0}' -f $_.VramMB } } }
    @{ n = 'vram max MB'; e = { if ($null -eq $_.VramMaxMB) { 'n/a' } else { '{0:N0}' -f $_.VramMaxMB } } }
    @{ n = 'note'; e = { $_.Note } }
)

$csv = Join-Path $repo "bench_results/bench-$(Get-Date -Format yyyyMMdd-HHmmss).csv"
New-Item -ItemType Directory -Force (Split-Path $csv) | Out-Null
$results | Export-Csv -NoTypeInformation -Path $csv
Write-Host "cpu % is of the whole machine ($cores logical cores); gpu % sums this process's GPU engines."
Write-Host "Saved $csv"
