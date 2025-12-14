# PowerShell helper to install minimal deps and download HateXplain dataset to data/hatexplain
# Run from the repository root in PowerShell (Windows)

python -m pip install --upgrade pip
python -m pip install datasets

# Run the downloader script
python .\src\fetch_and_save_dataset.py --output_dir .\data\hatexplain

Write-Host "If the script finished without errors, dataset is saved in .\data\hatexplain"