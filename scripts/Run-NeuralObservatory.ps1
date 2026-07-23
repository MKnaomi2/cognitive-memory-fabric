[CmdletBinding()]
param(
    [string]$Repository = 'C:\Hermes\hippocampal-memory',
    [string]$HermesHome = "$env:LOCALAPPDATA\hermes",
    [string]$RecordingsRoot = 'D:\HermesMemory\neural\recordings'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$python = Join-Path $Repository '.venv-neural\Scripts\python.exe'
$viewer = Join-Path $Repository 'viewer'
$stateDb = Join-Path $HermesHome 'state.db'
$logRoot = Join-Path $HermesHome 'logs\neural-observatory'
$npm = (Get-Command npm.cmd -ErrorAction Stop).Source

foreach ($required in @($python, $stateDb, (Join-Path $viewer 'package.json'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required observatory component is unavailable: $required"
    }
}
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Test-LoopbackPort {
    param([int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if ((Test-LoopbackPort 8765) -or (Test-LoopbackPort 3000)) {
    throw 'Observatory startup refused because port 8765 or 3000 is already in use.'
}

$apiArguments = @(
    '-m', 'hippocampal_memory.cli',
    '--home', $HermesHome,
    '--state-db', $stateDb,
    'observatory',
    '--recordings-root', $RecordingsRoot
)
$api = Start-Process `
    -FilePath $python `
    -ArgumentList $apiArguments `
    -WorkingDirectory $Repository `
    -RedirectStandardOutput (Join-Path $logRoot 'api.out.log') `
    -RedirectStandardError (Join-Path $logRoot 'api.err.log') `
    -WindowStyle Hidden `
    -PassThru

try {
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while (-not (Test-LoopbackPort 8765)) {
        if ($api.HasExited) {
            throw "Observatory API exited with code $($api.ExitCode)."
        }
        if ([DateTime]::UtcNow -ge $deadline) {
            throw 'Observatory API did not become ready within 30 seconds.'
        }
        Start-Sleep -Milliseconds 250
        $api.Refresh()
    }

    $ui = Start-Process `
        -FilePath $npm `
        -ArgumentList @('run', 'dev', '--', '--host', 'localhost') `
        -WorkingDirectory $viewer `
        -RedirectStandardOutput (Join-Path $logRoot 'viewer.out.log') `
        -RedirectStandardError (Join-Path $logRoot 'viewer.err.log') `
        -WindowStyle Hidden `
        -PassThru
    try {
        while (-not $api.HasExited -and -not $ui.HasExited) {
            Start-Sleep -Seconds 2
            $api.Refresh()
            $ui.Refresh()
        }
        if ($api.HasExited) {
            throw "Observatory API exited with code $($api.ExitCode)."
        }
        throw "Observatory viewer exited with code $($ui.ExitCode)."
    }
    finally {
        if (-not $ui.HasExited) {
            Stop-Process -Id $ui.Id -Force
        }
    }
}
finally {
    if (-not $api.HasExited) {
        Stop-Process -Id $api.Id -Force
    }
}
