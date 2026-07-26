[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$ValidationRoot,
    [string]$Python,
    [string]$ExpectedBranch = 'codex/continue-testing-v0.5.1'
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Root = [IO.Path]::GetFullPath($ValidationRoot).TrimEnd('\')
$Python = if ($Python) {
    [IO.Path]::GetFullPath($Python)
} else {
    Join-Path $Root 'venv-neural\Scripts\python.exe'
}
function Normalize-Path([string]$Path) {
    [IO.Path]::GetFullPath($Path).TrimEnd('\').ToLowerInvariant()
}

function Test-PathOverlap([string]$Left, [string]$Right) {
    $A = Normalize-Path $Left
    $B = Normalize-Path $Right
    return $A -eq $B -or $A.StartsWith($B + '\') -or $B.StartsWith($A + '\')
}

if (-not (Normalize-Path $Python).StartsWith((Normalize-Path $Root) + '\')) {
    throw "The validation Python environment must be below the validation root: $Python"
}

$LivePaths = @(
    'C:\Hermes\cognitive-memory-fabric',
    (Join-Path $env:LOCALAPPDATA 'hermes'),
    (Join-Path $env:LOCALAPPDATA 'hermes\state.db'),
    'C:\Hermes\Knowledge',
    'D:\HermesMemory\neural',
    'D:\HermesMemory\neural\recordings',
    'D:\HermesMemory\neural\checkpoints',
    (Join-Path $env:LOCALAPPDATA 'hermes\logs\neural-observatory')
)
$TestPaths = @(
    $Repo,
    (Join-Path $Root 'hermes-home'),
    (Join-Path $Root 'state\state.db'),
    (Join-Path $Root 'vault'),
    (Join-Path $Root 'recordings'),
    (Join-Path $Root 'checkpoints'),
    (Join-Path $Root 'logs'),
    (Join-Path $Root 'evaluation')
)
foreach ($TestPath in $TestPaths) {
    foreach ($LivePath in $LivePaths) {
        if (Test-PathOverlap $TestPath $LivePath) {
            throw "Refusing overlapping test/live paths: $TestPath <> $LivePath"
        }
    }
}

foreach ($Name in @('hermes-home', 'state', 'vault', 'recordings', 'checkpoints', 'logs', 'evaluation', 'evidence')) {
    New-Item -ItemType Directory -Path (Join-Path $Root $Name) -Force | Out-Null
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment is missing: $Python"
}
if ((git -C $Repo branch --show-current) -ne $ExpectedBranch) {
    throw "Validation must run from branch $ExpectedBranch"
}
if (git -C $Repo status --porcelain) {
    throw 'Validation must run from a clean committed branch state'
}
$TestedCommit = git -C $Repo rev-parse HEAD

function Get-HermesTaskFingerprint {
    $Definitions = @(Get-ScheduledTask |
        Where-Object { $_.TaskName -like 'Hermes_*' } |
        Sort-Object TaskPath, TaskName |
        ForEach-Object {
            "$($_.TaskPath)$($_.TaskName)`n" +
                (Export-ScheduledTask -TaskName $_.TaskName -TaskPath $_.TaskPath)
        }) -join "`n"
    $Bytes = [Text.Encoding]::UTF8.GetBytes($Definitions)
    [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory)][string]$File,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][string]$Log,
        [string]$WorkingDirectory = $Repo
    )
    Push-Location $WorkingDirectory
    try {
        & $File @Arguments 2>&1 | Tee-Object -FilePath $Log
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed ($LASTEXITCODE): $File $($Arguments -join ' ')"
        }
    } finally {
        Pop-Location
    }
}

$TaskFingerprintBefore = Get-HermesTaskFingerprint
$env:HIPPOCAMPAL_MEMORY_HOME = Join-Path $Root 'hermes-home'
$env:CMF_E2E_ROOT = Join-Path $Root 'e2e'
$env:CMF_E2E_API_PORT = '8766'
$env:CMF_E2E_PYTHON = $Python
$env:NEXT_PUBLIC_OBSERVATORY_API_ORIGIN = 'http://127.0.0.1:8766'

& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader |
    Set-Content -LiteralPath (Join-Path $Root 'evidence\gpu.txt')
Invoke-Checked $Python @('-c', 'import json,torch; print(json.dumps({"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))') (Join-Path $Root 'evidence\torch.json')
$Torch = Get-Content -Raw (Join-Path $Root 'evidence\torch.json') | ConvertFrom-Json
if (-not $Torch.cuda_available) {
    throw 'torch.cuda.is_available() is false'
}

$PytestXml = Join-Path $Root 'evidence\pytest.xml'
Invoke-Checked $Python @('-m', 'pytest', '-q', '--junitxml', $PytestXml) (Join-Path $Root 'logs\pytest.log')
Invoke-Checked $Python @('scripts\assert_pytest_counts.py', $PytestXml, '--expected', '65') (Join-Path $Root 'logs\pytest-counts.log')
Invoke-Checked $Python @('-m', 'ruff', 'check', 'src', 'tests', 'integrations', 'scripts') (Join-Path $Root 'logs\ruff.log')
Invoke-Checked $Python @('-m', 'compileall', '-q', 'src', 'integrations', 'scripts', 'tests') (Join-Path $Root 'logs\compileall.log')
Invoke-Checked $Python @('tests\observatory_integration.py') (Join-Path $Root 'logs\observatory-integration.log')

$CircuitLog = Join-Path $Root 'evidence\circuit-check.json'
Invoke-Checked (Join-Path (Split-Path $Python) 'cognitive-memory.exe') @('circuit-check', '--device', 'cuda') $CircuitLog
$Circuit = Get-Content -Raw $CircuitLog | ConvertFrom-Json
if (
    $Circuit.status -ne 'completed' -or
    $Circuit.device -notlike 'cuda*' -or
    $Circuit.neurons -ne 36864 -or
    $Circuit.synapses -ne 770048 -or
    $Circuit.engram_neurons -le 0 -or
    $Circuit.time_cells -le 0
) {
    throw "Circuit acceptance contract failed: $($Circuit | ConvertTo-Json -Compress)"
}

Invoke-Checked $Python @('scripts\engineering_validation.py', '--root', $Root, '--device', 'cuda') (Join-Path $Root 'logs\engineering-domain.log')

$Viewer = Join-Path $Repo 'viewer'
Invoke-Checked 'npm.cmd' @('ci') (Join-Path $Root 'logs\npm-ci.log') $Viewer
Invoke-Checked 'npm.cmd' @('audit', '--audit-level=high') (Join-Path $Root 'logs\npm-audit.log') $Viewer
Invoke-Checked 'npm.cmd' @('run', 'lint') (Join-Path $Root 'logs\npm-lint.log') $Viewer
Invoke-Checked 'npm.cmd' @('run', 'build') (Join-Path $Root 'logs\npm-build.log') $Viewer
Invoke-Checked 'npm.cmd' @('test') (Join-Path $Root 'logs\npm-test.log') $Viewer
Invoke-Checked 'npx.cmd' @('playwright', 'install', 'chromium') (Join-Path $Root 'logs\playwright-install.log') $Viewer
Invoke-Checked 'npm.cmd' @('run', 'e2e') (Join-Path $Root 'logs\playwright.log') $Viewer

$Cli = Join-Path (Split-Path $Python) 'cognitive-memory.exe'
foreach ($Split in @('development', 'holdout')) {
    Invoke-Checked $Cli @('evaluate', 'verify', "benchmarks\results\v0.5.1\$Split") (Join-Path $Root "logs\verify-committed-$Split.log")
    $Output = Join-Path $Root "evaluation\$Split"
    Invoke-Checked $Cli @(
        'evaluate', 'run',
        '--profile', $Split,
        '--output', $Output,
        '--conditions', 'fabric-symbolic', 'fabric-neural',
        '--neural-weight', '0.05',
        '--neural-margin-min', '0.0',
        '--neural-activation-min', '0.70'
    ) (Join-Path $Root "logs\reproduce-$Split.log")
    Invoke-Checked $Cli @('evaluate', 'verify', $Output) (Join-Path $Root "logs\verify-reproduced-$Split.log")
    $Manifest = Get-Content -Raw (Join-Path $Output 'manifest.json') | ConvertFrom-Json
    if ($Manifest.git.dirty -or $Manifest.git.commit -ne $TestedCommit) {
        throw "$Split manifest does not identify the clean tested commit"
    }
    if (
        $Manifest.configuration.neural_weight -ne 0.05 -or
        $Manifest.configuration.neural_margin_min -ne 0.0 -or
        $Manifest.configuration.neural_activation_min -ne 0.70 -or
        ($Manifest.conditions -join ',') -ne 'fabric-symbolic,fabric-neural'
    ) {
        throw "$Split manifest does not contain the frozen candidate parameters"
    }
}
$Candidate = Get-Content -Raw (Join-Path $Repo 'benchmarks\neural-candidate-v0.5.1.json') | ConvertFrom-Json
if ($Candidate.cue_mode -ne 'lexical') {
    throw 'Frozen neural candidate does not identify lexical cue mode'
}

$TaskFingerprintAfter = Get-HermesTaskFingerprint
if ($TaskFingerprintAfter -ne $TaskFingerprintBefore) {
    throw 'Production Hermes scheduled-task definitions changed during validation'
}
[ordered]@{
    status = 'completed'
    tested_commit = $TestedCommit
    validation_root = $Root
    overlap_check = 'passed'
    torch = $Torch
    circuit = $Circuit
    scheduled_tasks_unchanged = $true
    ports = @{ api = 8766; viewer = 5173 }
    evaluation = @{
        cue_mode = 'lexical'
        neural_weight = 0.05
        neural_margin_min = 0.0
        neural_activation_min = 0.70
        conditions = @('fabric-symbolic', 'fabric-neural')
        holdout_use = 'reproducibility evidence only'
    }
} | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $Root 'evidence\validation-summary.json')

Write-Host "Engineering validation completed for $TestedCommit"
