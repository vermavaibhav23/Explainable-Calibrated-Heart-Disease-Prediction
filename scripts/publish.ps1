param([string]$Name = 'Explainable-Calibrated-Heart-Disease-Prediction')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
gh auth status
if ($LASTEXITCODE -ne 0) { throw 'Sign in with gh auth login, then run this script again.' }
git rev-parse --is-inside-work-tree
if ($LASTEXITCODE -ne 0) { throw 'Initialize and commit the local repository first.' }
$remote = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0 -and $remote) {
    git push -u origin main
} else {
    gh repo create $Name --public --source . --remote origin --push --description 'Explainable and calibrated heart disease classification with nested validation, SHAP and Streamlit'
}
if ($LASTEXITCODE -ne 0) { throw 'Repository publishing failed; inspect the error above before retrying.' }

