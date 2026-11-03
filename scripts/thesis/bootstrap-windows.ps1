# Downloads a fixed official portable TinyTeX snapshot. No PATH/registry changes.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$lock = Get-Content -Raw -Encoding UTF8 (Join-Path $root 'config/thesis_environment.lock.json') | ConvertFrom-Json
$tools = Join-Path $root '.tools/thesis'
New-Item -ItemType Directory -Force -Path $tools | Out-Null
$archive = Join-Path $tools 'TinyTeX-v2026.10.zip'
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri $lock.windows_tinytex_url -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $lock.windows_tinytex_sha256) {
    throw 'TinyTeX SHA256 mismatch. Do not run the downloaded files.'
}
$runtime = Join-Path $tools 'TinyTeX'
if (-not (Test-Path -LiteralPath (Join-Path $runtime '.snapshot-sha256'))) {
    python (Join-Path $PSScriptRoot 'extract-runtime.py')
    if ($LASTEXITCODE -ne 0) { throw 'Fixed TinyTeX snapshot extraction failed.' }
}
$env:PATH = (Join-Path $runtime 'bin/windows') + ';' + $env:PATH
$env:LC_ALL='C'; $env:LC_CTYPE='C'; $env:LANG='C'
python (Join-Path $PSScriptRoot 'install-runtime-packages.py')
if ($LASTEXITCODE -ne 0) { throw 'Fixed supplementary package installation failed.' }
Write-Host 'Portable runtime ready. Next: scripts/thesis/prepare-fonts.ps1 and scripts/thesis/build.ps1'
