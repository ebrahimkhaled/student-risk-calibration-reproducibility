# Step 6: every year-on-year pair, both feature sets: fit, Check A, Check B.
#
# Resumable: a step whose output already exists is skipped, so the run can be stopped and
# restarted at any point without redoing finished work. Each step's console output goes to
# logs\<set>_<pair>_<step>.log, and a one-line status per step to logs\run_all.tsv.
#
#   powershell -ExecutionPolicy Bypass -File 06_run_all.ps1
#   powershell -ExecutionPolicy Bypass -File 06_run_all.ps1 -Sets stable

param([string[]]$Sets = @("stable", "full"))

$ErrorActionPreference = "Continue"
$Project = if ($env:ICT_ROOT) { $env:ICT_ROOT } else { "D:\Research\paper_ict_student_risk" }
$Engine  = if ($env:TFM_REBUILD_ROOT) { $env:TFM_REBUILD_ROOT } else { "D:\Research\paper_tfm_calibration\rebuild" }
. "$Engine\env.ps1"
$Py   = "$Engine\.venv-gpu\Scripts\python.exe"
$Code = "$Project\code"
$Logs = "$Project\logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
$Status = "$Logs\run_all.tsv"
if (-not (Test-Path $Status)) { "time`tset`tpair`tstep`texit`tseconds" | Out-File $Status -Encoding utf8 }

$Pairs = Import-Csv "$Project\data\scope\year_on_year_pairs.csv"

function Step($set, $pair, $name, $done, [scriptblock]$cmd) {
    if (Test-Path $done) { Write-Host "  skip  $set $pair $name (done)"; return $true }
    $log = "$Logs\${set}_${pair}_${name}.log"
    $t0 = Get-Date
    & $cmd *> $log
    $code = $LASTEXITCODE
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    "$((Get-Date).ToString('s'))`t$set`t$pair`t$name`t$code`t$secs" | Out-File $Status -Append -Encoding utf8
    $ok = ($code -eq 0) -and (Test-Path $done)
    Write-Host ("  {0,-5} {1} {2} {3}  {4}s" -f ($(if ($ok) {"ok"} else {"FAIL"})), $set, $pair, $name, $secs)
    return $ok
}

foreach ($set in $Sets) {
    Write-Host "=== feature set: $set ==="
    foreach ($p in $Pairs) {
        $pair = "$($p.module)_$($p.train)_$($p.test)"
        $dir = "$Project\results\$set\$pair"
        $ok = Step $set $pair "fit" "$dir\metrics.csv" {
            & $Py "$Code\03_fit_pair.py" $p.module $p.train $p.test --features $set --device cuda }
        if (-not $ok) { continue }      # nothing to check without predictions
        Step $set $pair "check_a" "$dir\check_a.csv" {
            & Rscript "$Code\04_check_a.R" $set $pair } | Out-Null
        Step $set $pair "check_b" "$dir\check_b.csv" {
            & Rscript "$Code\05_check_b.R" $set $pair } | Out-Null
    }
}
Write-Host "finished; status in $Status"
