param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
function Full([string]$Relative) { [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Relative)) }
function Assert-InRepo([string]$Path) { if (-not $Path.StartsWith($RepoRoot)) { throw "Path outside repo: $Path" } }
function Move-Safe([string]$SourceRel,[string]$DestinationRel) {
    $source=Full $SourceRel; $destination=Full $DestinationRel
    Assert-InRepo $source; Assert-InRepo $destination
    if (-not (Test-Path -LiteralPath $source)) { Write-Host "[skip missing] $SourceRel"; return }
    Write-Host "[move] $SourceRel -> $DestinationRel"
    if ($Apply) {
        $parent=Split-Path -Parent $destination
        New-Item -ItemType Directory -Force $parent | Out-Null
        Move-Item -LiteralPath $source -Destination $destination
    }
}

$dirs=@(
 'tasks/task1/assets','tasks/task1/predictions','tasks/task2/predictions','tasks/task3/predictions',
 'tasks/task1/experiment_record/legacy-saves/checkpoints',
 'tasks/task2/experiment_record/legacy-saves/checkpoints',
 'tasks/task3/experiment_record/legacy-saves/checkpoints',
 'tasks/common/reports','历史回滚代码/root_compat_interfaces_20260817','历史回滚代码/root_duplicate_reports_20260817'
)
if($Apply){ foreach($d in $dirs){ New-Item -ItemType Directory -Force (Full $d) | Out-Null } }

Move-Safe 'tasks/task1/reports' 'tasks/task1/report'
Move-Safe 'tasks/task2/reports' 'tasks/task2/report'
Move-Safe 'tasks/task3/reports' 'tasks/task3/report'
Move-Safe 'frame_prototypes' 'tasks/task1/assets/frame_prototypes'
if(Test-Path -LiteralPath (Full 'predictions')){
  Get-ChildItem -LiteralPath (Full 'predictions') -Force | ForEach-Object {
    Move-Safe ("predictions/"+$_.Name) ("tasks/task2/predictions/"+$_.Name)
  }
}
Move-Safe 'saves/model_task1_best.bin' 'tasks/task1/experiment_record/legacy-saves/checkpoints/model_task1_best.bin'
Move-Safe 'saves/model_task2_best.bin' 'tasks/task2/experiment_record/legacy-saves/checkpoints/model_task2_best.bin'
Move-Safe 'saves/model_task3_best.bin' 'tasks/task3/experiment_record/legacy-saves/checkpoints/model_task3_best.bin'
Move-Safe 'base_result.txt' 'tasks/common/reports/base_result.txt'
Move-Safe 'ls.ipynb' '历史回滚代码/ls.ipynb'

$wrappers=@(
'analyze_task1_ensemble.py','analyze_task1_prototype_hybrid.py','analyze_task2_threshold.py',
'build_frame_prototypes.py','check_frame_prototypes.py','check_task1_markers.py',
'dataset_task1.py','dataset_task2.py','dataset_task3.py','experiment_utils.py','frame_prototype_utils.py',
'model_task1_prototype.py','model_task1.py','model_task2.py','model_task3.py','params.py',
'predict_task1_ensemble.py','predict_task1_prototype_hybrid.py','predict_task1.py','predict_task2.py','predict_task3.py',
'refresh_project_index.py','smoke_task1_marker_forward.py','smoke_task2_experiments.py',
'train_task1_prototype.py','train_task1.py','train_task2.py','train_task3.py','validate_task2_data.py'
)
foreach($f in $wrappers){ Move-Safe $f ("历史回滚代码/root_compat_interfaces_20260817/"+$f) }
$reports=@('experiment_analysis_2026-08-17.md','frame_prototype_experiment_2026-08-17.md','model1_optimization_plan.md','target_marker_experiment_2026-08-17.md','task1_experiment_report_2026-08-17.md')
foreach($f in $reports){ Move-Safe $f ("历史回滚代码/root_duplicate_reports_20260817/"+$f) }

if($Apply){
  foreach($emptyDir in @('predictions','saves')){
    $fullDir=Full $emptyDir
    if((Test-Path -LiteralPath $fullDir) -and ((Get-ChildItem -LiteralPath $fullDir -Force | Measure-Object).Count -eq 0)){
      Remove-Item -LiteralPath $fullDir
      Write-Host "[removed empty root directory] $emptyDir"
    }
  }
}

if(-not $Apply){ Write-Host "`nDry run only. Re-run with -Apply to perform the moves." }
else { Write-Host "`nMigration completed. Root code interfaces are archived; use python -m tasks.taskX..." }



