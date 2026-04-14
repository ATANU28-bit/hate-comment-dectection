# PowerShell script to download Jigsaw dataset using Kaggle CLI.
# Requires: `pip install kaggle` and setting up `~/.kaggle/kaggle.json` credentials.

param(
    [string]$outputDir = "..\data\jigsaw"
)

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Error "kaggle CLI not found. Install with: pip install kaggle"
    exit 1
}

mkdir -Force $outputDir | Out-Null
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p $outputDir
Write-Host "Downloaded to $outputDir"