# Overnight chain, most important results first:
#   1. the main analysis (stable feature set, all nine pairs)
#   2. the shrinkage plasmode for Figure 7 (needs Check B of BBB 2013J -> 2014J from step 1)
#   3. the secondary analysis (full feature set)
# Each step is resumable on its own, so rerunning this script skips finished work.

$Root = if ($env:ICT_ROOT) { $env:ICT_ROOT } else { "D:\Research\paper_ict_student_risk" }
$Code = "$Root\code"
$Logs = "$Root\logs"
New-Item -ItemType Directory -Force $Logs | Out-Null

"$((Get-Date).ToString('s'))  start: stable set" | Out-File "$Logs\overnight.txt" -Append -Encoding utf8
& powershell -ExecutionPolicy Bypass -File "$Code\06_run_all.ps1" -Sets stable

"$((Get-Date).ToString('s'))  start: shrinkage plasmode" | Out-File "$Logs\overnight.txt" -Append -Encoding utf8
& Rscript "$Code\07_plasmode_shrinkage.R" *> "$Logs\plasmode.log"
"$((Get-Date).ToString('s'))  plasmode exit $LASTEXITCODE" | Out-File "$Logs\overnight.txt" -Append -Encoding utf8

"$((Get-Date).ToString('s'))  start: full set" | Out-File "$Logs\overnight.txt" -Append -Encoding utf8
& powershell -ExecutionPolicy Bypass -File "$Code\06_run_all.ps1" -Sets full

"$((Get-Date).ToString('s'))  all done" | Out-File "$Logs\overnight.txt" -Append -Encoding utf8
