# Local stand-in for the `gpu` job in .github/workflows/ci.yml.
#
# That job needs a self-hosted runner, which needs repo admin to register. Until
# then this runs the identical steps on your machine. Keep the two in sync: if
# you change a step here, change it there.
#
#   pwsh -File scripts/gpu-check.ps1

$ErrorActionPreference = 'Stop'

# The smoke tests launch the built .exe directly, which never goes through cargo,
# so .cargo/config.toml's [env] does not apply. Without validation on, the release
# window example hits the known async-UAF driver crash (src/lib.rs:1168).
$env:SUNRAY_ENABLE_VALIDATION_LAYER = '1'
$env:SUNRAY_SERIALIZE_FRAMES = '1'

# Run from the repo root: the examples resolve examples/log4rs.yaml and the .glb
# against the current directory.
Set-Location (Join-Path $PSScriptRoot '..')

function Invoke-Step($Name, $ScriptBlock) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $ScriptBlock
    if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit $LASTEXITCODE)" }
}

# Two conditions, because surviving is not the same as working:
#
#   1. It must not exit inside $Seconds. A healthy window never exits on its own,
#      so an early exit means a panic or the graceful error path (handle_srresult
#      exits the event loop on any non-OUT_OF_DATE error).
#   2. It must print $Heartbeat. A deadlocked renderer also never exits, so
#      condition 1 alone reports a hung binary as healthy -- that is not
#      hypothetical, it is how a swapchain-init deadlock once passed this check.
#      The window example prints "[heartbeat] frame N fps F" once a second, which
#      only happens if frames are actually being submitted.
#
# $Heartbeat is empty for bevy_app, which has no such output; that check stays
# liveness-only and is non-fatal anyway.
function Test-Liveness($Exe, $Seconds = 10, $Heartbeat = $null) {
    $out = [System.IO.Path]::GetTempFileName()
    try {
        $p = Start-Process -FilePath $Exe -WorkingDirectory $PWD -PassThru -RedirectStandardOutput $out
        if ($p.WaitForExit($Seconds * 1000)) {
            throw "$Exe exited early (code $($p.ExitCode))"
        }
        $p.Kill(); $p.WaitForExit()

        if ($Heartbeat) {
            $beats = @(Select-String -Path $out -Pattern $Heartbeat -SimpleMatch)
            if ($beats.Count -lt 2) {
                throw "$Exe stayed alive but only produced $($beats.Count) '$Heartbeat' line(s) in ${Seconds}s -- it is running but not rendering (deadlock?)"
            }
            Write-Host "$Exe rendered for ${Seconds}s ($($beats.Count) heartbeats)" -ForegroundColor Green
        } else {
            Write-Host "$Exe survived ${Seconds}s" -ForegroundColor Green
        }
    } finally {
        Remove-Item $out -ErrorAction SilentlyContinue
    }
}

Invoke-Step 'Test (incl. GPU)' { cargo test --release -- --include-ignored }

Invoke-Step 'Render offscreen' { cargo png }

Write-Host "`n=== Compare against baseline ===" -ForegroundColor Cyan
$want = (Get-Content examples/png/render.sha256).Trim()
$got = (Get-FileHash examples/png/render.png -Algorithm SHA256).Hash
if ($got -ne $want) {
    throw "render changed: want=$want got=$got (re-baseline if intended)"
}
Write-Host "pixel-identical ($got)" -ForegroundColor Green

Write-Host "`n=== Smoke-test window example (10s) ===" -ForegroundColor Cyan
Invoke-Step 'Build window example' { cargo build --release --example window }
Test-Liveness 'target\release\examples\window.exe' -Heartbeat '[heartbeat]'

# Optional, matching `continue-on-error: true` in the workflow: reported but never
# fatal, so a moving bevy integration can't block everything else.
Write-Host "`n=== Smoke-test bevy example (10s, optional) ===" -ForegroundColor Cyan
try {
    Invoke-Step 'Build bevy example' { cargo build --release --features bevy --example bevy_app }
    Test-Liveness 'target\release\examples\bevy_app.exe'
} catch {
    Write-Host "bevy check failed (non-fatal): $_" -ForegroundColor Yellow
}

Write-Host "`nAll GPU checks passed." -ForegroundColor Green
