[CmdletBinding()]
param(
    [string]$Repository = 'C:\Hermes\cognitive-memory-fabric',
    [string]$HermesHome = "$env:LOCALAPPDATA\hermes",
    [int]$Port = 8767,
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cuda'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$python = Join-Path $Repository '.venv-neural\Scripts\python.exe'
$stateDb = Join-Path $HermesHome 'state.db'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'The isolated neural Python environment is unavailable.'
}
if (-not (Test-Path -LiteralPath $stateDb -PathType Leaf)) {
    throw 'The Hermes state database is unavailable.'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Neural readout startup refused because port $Port is already in use."
}

& $python -m hippocampal_memory.cli `
    --home $HermesHome `
    --state-db $stateDb `
    neural-serve `
    --port $Port `
    --device $Device `
    --cue-mode lexical `
    --neural-weight 0.05 `
    --neural-margin-min 0.0 `
    --neural-activation-min 0.7
exit $LASTEXITCODE
