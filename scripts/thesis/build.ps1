param([ValidateSet('local','docker')][string]$Backend='local',
      [ValidateSet('strict','original','preview')][string]$Fonts='strict',
      [ValidateSet('main','results')][string]$Document='main')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$portableBin = Join-Path $root '.tools/thesis/TinyTeX/bin/windows'
if (Test-Path -LiteralPath $portableBin) { $env:PATH = $portableBin + ';' + $env:PATH }
$env:LC_ALL='C'; $env:LC_CTYPE='C'; $env:LANG='C'
python (Join-Path $PSScriptRoot 'thesis.py') build --backend $Backend --fonts $Fonts --document $Document
exit $LASTEXITCODE
