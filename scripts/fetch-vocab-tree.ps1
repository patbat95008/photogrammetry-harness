# Fetch the COLMAP vocabulary tree used for loop detection in sequential matching.
#
# READ THIS BEFORE RUNNING IT.
#
# The trees published at demuc.de are the legacy FLANN format. COLMAP replaced FLANN
# with faiss for its visual index in May 2025, and the build in tools/ (4.2.0, commit
# be5e291) cannot read the old format. It does not reject it either: it aborts with
# STATUS_STACK_BUFFER_OVERRUN (0xC0000409) partway through matching, after feature
# extraction has already been paid for.
#
# The harness checks the format before use (server/pgh/stages/sparse.py,
# vocab_tree_status) and falls back to exhaustive matching, so a legacy tree is
# harmless -- it is simply never used. The doctor page reports which you have.
#
# At the time of writing no faiss-format tree has been published. When one is, point
# $Url at it; the four-byte version check will start returning "ok" on its own.
#
# Exhaustive matching needs no tree at all and cannot miss a loop, because it compares
# every pair. It is the right choice below roughly 500 images -- 183 frames of the cup
# orbit matched exhaustively in 6.3 minutes on the 4090.

$ErrorActionPreference = "Stop"

$Url = "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"
$DataDir = Join-Path $PSScriptRoot "..\data"
$Target = Join-Path $DataDir "vocab_tree_flickr100K_words32K.bin"

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

if (Test-Path $Target) {
    Write-Host "Already present: $Target"
} else {
    Write-Host "Downloading $Url"
    # Download to a temporary name and rename, so an interrupted transfer never
    # leaves a half-written file that looks like a usable tree.
    $Partial = "$Target.part"
    Invoke-WebRequest -Uri $Url -OutFile $Partial -UseBasicParsing
    Move-Item -Force $Partial $Target
    Write-Host "Saved to $Target"
}

# Four bytes are enough to tell the formats apart: the faiss-era file opens with a
# version field of 1 or 2, a FLANN tree with its word count (32762 for this one).
$Bytes = [System.IO.File]::ReadAllBytes($Target)[0..3]
$Version = [System.BitConverter]::ToUInt32($Bytes, 0)

if ($Version -eq 1 -or $Version -eq 2) {
    Write-Host "Format: faiss (version $Version) - loop detection is available."
} else {
    Write-Host ""
    Write-Warning "Format: legacy FLANN (version field reads $Version)."
    Write-Warning "This COLMAP cannot read it. The align stage will match exhaustively"
    Write-Warning "instead, which is correct and, at these image counts, barely slower."
}
