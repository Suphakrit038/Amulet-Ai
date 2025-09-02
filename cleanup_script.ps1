# สคริปต์ลบไฟล์ที่ไม่จำเป็น

Write-Host "🗑️ Starting cleanup process..."

# ไฟล์ที่จะลบ
$filesToDelete = @(
    "analyze_dataset.py",
    "app.py", 
    "check_data_models.py",
    "cleanup_files.bat",
    "complete_organizer.py",
    "config.json",
    "dataset_inspector.py", 
    "dataset_organizer.py",
    "debug_copy.py",
    "organize_dataset.bat",
    "organize_dataset.ps1", 
    "organize_dataset.py",
    "organize_karaoke.bat",
    "organize_step1.py",
    "quick_dataset_stats.py",
    "rename_dataset_files.py",
    "requirements.txt",
    "simple_copy.py",
    "simple_organizer.py",
    "test_copy.bat",
    "test_copy.py"
)

# ไฟล์ .md ที่จะลบ (ยกเว้น README.md)
$mdFilesToDelete = @(
    "BUGFIXES_SUMMARY.md",
    "CLEANUP_REPORT.md", 
    "COMPLETE_DATASET_INSPECTION.md",
    "DATASET_INSPECTION_REPORT.md",
    "DATASET_ORGANIZATION_GUIDE.md",
    "DATASET_ORGANIZATION_STATUS.md",
    "DATA_MODEL_ANALYSIS_REPORT.md",
    "KARAOKE_DATASET_ORGANIZATION.md",
    "PROJECT_STRUCTURE.md"
)

# ลบไฟล์
foreach($file in $filesToDelete) {
    if(Test-Path $file) {
        try {
            Remove-Item $file -Force
            Write-Host "✅ Deleted: $file"
        } catch {
            Write-Host "❌ Failed to delete: $file - $($_.Exception.Message)"
        }
    }
}

foreach($file in $mdFilesToDelete) {
    if(Test-Path $file) {
        try {
            Remove-Item $file -Force
            Write-Host "✅ Deleted: $file"
        } catch {
            Write-Host "❌ Failed to delete: $file - $($_.Exception.Message)"
        }
    }
}

# ลบโฟลเดอร์
$foldersToDelete = @(
    "data-processing",
    "dev-tools", 
    "logs",
    ".pytest_cache"
)

foreach($folder in $foldersToDelete) {
    if(Test-Path $folder) {
        try {
            Remove-Item $folder -Recurse -Force
            Write-Host "✅ Deleted folder: $folder"
        } catch {
            Write-Host "❌ Failed to delete folder: $folder - $($_.Exception.Message)"
        }
    }
}

# ลบ __pycache__ folders
$pycacheFolders = Get-ChildItem -Recurse -Directory -Name "__pycache__"
foreach($folder in $pycacheFolders) {
    try {
        Remove-Item $folder -Recurse -Force
        Write-Host "✅ Deleted __pycache__: $folder"
    } catch {
        Write-Host "❌ Failed to delete __pycache__: $folder - $($_.Exception.Message)"
    }
}

Write-Host "🎉 Cleanup completed!"
