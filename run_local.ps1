param(
    [switch]$SkipInstall,
    [switch]$ForceTrain
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pip = Join-Path $projectRoot ".venv\Scripts\pip.exe"

if (-not $SkipInstall) {
    Write-Host "Installing dependencies..."
    & $pip install -r requirements.txt
}

$requiredModels = @(
    "models\diabetes_model.pkl",
    "models\diabetes_scaler.pkl",
    "models\diabetes_features.pkl",
    "models\heart_model.pkl",
    "models\heart_scaler.pkl",
    "models\heart_features.pkl",
    "models\cancer_model.pkl",
    "models\cancer_scaler.pkl",
    "models\cancer_features.pkl",
    "models\kidney_model.pkl",
    "models\kidney_scaler.pkl",
    "models\kidney_features.pkl",
    "models\liver_model.pkl",
    "models\liver_scaler.pkl",
    "models\liver_features.pkl",
    "models\stroke_model.pkl",
    "models\stroke_scaler.pkl",
    "models\stroke_features.pkl",
    "models\mental_health_model.pkl",
    "models\mental_health_scaler.pkl",
    "models\mental_health_features.pkl"
)

$missingModels = $requiredModels | Where-Object { -not (Test-Path $_) }
if ($ForceTrain -or $missingModels.Count -gt 0) {
    Write-Host "Training models..."
    & $python "src\train.py"
}

Write-Host "Initializing database..."
& $python -c "from app import init_db; init_db(); print('Database ready')"

Write-Host "Starting Flask app at http://127.0.0.1:5000"
& $python "app.py"
