[CmdletBinding()]
param(
    [string]$Repository = 'C:\Hermes\cognitive-memory-fabric',
    [string]$HermesHome = "$env:LOCALAPPDATA\hermes",
    [string]$StateRoot = 'D:\HermesMemory\neural',
    [int]$MaxMemories = 8
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

$arguments = @(
    '-m', 'hippocampal_memory.cli',
    '--home', $HermesHome,
    '--state-db', $stateDb,
    'sleep',
    '--state-root', $StateRoot,
    '--max-memories', [string]$MaxMemories
)
& $python @arguments
exit $LASTEXITCODE
