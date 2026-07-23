[CmdletBinding()]
param(
    [string]$TaskName = 'Hermes_Hippocampal_Sleep',
    [string]$Repository = 'C:\Hermes\cognitive-memory-fabric',
    [switch]$Enable
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$runner = Join-Path $Repository 'scripts\Run-HippocampalSleep.ps1'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw 'The sleep runner is unavailable.'
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runner`""
$trigger = New-ScheduledTaskTrigger -Daily -At '2:10 AM'
$settings = New-ScheduledTaskSettingsSet `
    -RunOnlyIfIdle `
    -IdleDuration (New-TimeSpan -Minutes 30) `
    -IdleWaitTimeout (New-TimeSpan -Hours 3) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew `
    -WakeToRun
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
    -Description 'Exclusive-GPU NREM/REM replay for provenance-aware Hermes memory.' `
    -Force | Out-Null

if (-not $Enable) {
    Disable-ScheduledTask -TaskName $TaskName | Out-Null
}
Get-ScheduledTask -TaskName $TaskName |
    Select-Object TaskName, State, @{Name='Enabled';Expression={$_.Settings.Enabled}}
