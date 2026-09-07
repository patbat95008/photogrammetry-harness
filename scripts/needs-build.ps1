# Decide whether the frontend needs rebuilding.
#
# Exit code 1 = rebuild needed, 0 = the build is current.
#
# This lives in its own file rather than inline in launch.bat because a pipeline
# inside a batch `for /f (...)` has to survive two levels of escaping, and getting
# it wrong fails silently-ish: cmd hands PowerShell a literal `^|` and the check
# errors out instead of answering.

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$built = Join-Path $root 'web\dist\index.html'

if (-not (Test-Path $built)) {
    Write-Host '  No build found.'
    exit 1
}

$sources = @(
    (Join-Path $root 'web\src'),
    (Join-Path $root 'web\index.html'),
    (Join-Path $root 'web\package.json'),
    (Join-Path $root 'web\vite.config.ts')
) | Where-Object { Test-Path $_ }

if (-not $sources) { exit 0 }

$builtAt = (Get-Item $built).LastWriteTimeUtc
$newest = Get-ChildItem -Path $sources -Recurse -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1

if ($newest -and $newest.LastWriteTimeUtc -gt $builtAt) {
    Write-Host "  Sources changed since the last build ($($newest.Name))."
    exit 1
}

exit 0
