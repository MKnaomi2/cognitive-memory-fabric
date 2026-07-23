[CmdletBinding()]
param(
    [string]$TaskName = 'Hermes_Neural_Observatory',
    [string]$Repository = 'C:\Hermes\cognitive-memory-fabric',
    [switch]$Enable
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$runner = Join-Path $Repository 'scripts\Run-NeuralObservatory.ps1'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw 'The neural observatory runner is unavailable.'
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runner`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description 'Loopback-only Hermes neural telemetry API and 3-D observatory.' `
    -Force | Out-Null

if (-not $Enable) {
    Disable-ScheduledTask -TaskName $TaskName | Out-Null
}
Get-ScheduledTask -TaskName $TaskName |
    Select-Object TaskName, State, @{Name='Enabled';Expression={$_.Settings.Enabled}}
