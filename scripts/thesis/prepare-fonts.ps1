# Windows only. Copy installed fonts into this private project; no system changes.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dest = Join-Path $root 'thesis/latex/fonts'
$kaiTarget = Join-Path $dest 'kaiu.ttf'
if (-not (Test-Path -LiteralPath $kaiTarget)) {
    $kaiSource = Join-Path $env:WINDIR 'Fonts/kaiu.ttf'
    if (-not (Test-Path -LiteralPath $kaiSource)) { throw 'Supply a legally usable kaiu.ttf in thesis/latex/fonts first.' }
    Copy-Item -LiteralPath $kaiSource -Destination $kaiTarget
}
$kaiTarget = Join-Path $dest 'kaiu.ttf'
if (-not (Test-Path -LiteralPath $kaiTarget)) {
    $kaiSource = Join-Path $env:WINDIR 'Fonts/kaiu.ttf'
    if (-not (Test-Path -LiteralPath $kaiSource)) { throw 'Supply a legally usable kaiu.ttf in thesis/latex/fonts first.' }
    Copy-Item -LiteralPath $kaiSource -Destination $kaiTarget
}
foreach ($name in @('times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf')) {
    $source = Join-Path $env:WINDIR ('Fonts/' + $name)
    $target = Join-Path $dest $name
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing installed font: $source" }
    if (Test-Path -LiteralPath $target) {
        if ((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash) {
            throw "Different font already exists; keep it and review manually: $target"
        }
    } else { Copy-Item -LiteralPath $source -Destination $target }
}
Write-Host 'Fonts prepared for private use. Check licensing before redistributing.'
