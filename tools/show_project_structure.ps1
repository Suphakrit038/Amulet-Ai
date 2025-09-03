<#
.SYNOPSIS
    แสดงโครงสร้างไฟล์โปรเจค Amulet-AI ในรูปแบบที่อ่านง่าย
.DESCRIPTION
    สคริปต์นี้จะแสดงโครงสร้างไฟล์และโฟลเดอร์ของโปรเจค Amulet-AI
    โดยสามารถกรองประเภทไฟล์และความลึกของโครงสร้างได้
.PARAMETER Depth
    ความลึกสูงสุดของโครงสร้างไฟล์ที่จะแสดง (ค่าเริ่มต้น: 4)
.PARAMETER ExcludeFiles
    ไม่แสดงไฟล์ แสดงเฉพาะโฟลเดอร์
.PARAMETER ExcludePattern
    รูปแบบของไฟล์หรือโฟลเดอร์ที่ต้องการข้าม (เช่น *.pyc, __pycache__)
.PARAMETER OutputFile
    บันทึกผลลัพธ์ลงในไฟล์ที่ระบุ
.EXAMPLE
    .\show_project_structure.ps1 -Depth 3
    แสดงโครงสร้างไฟล์ลึกสุด 3 ระดับ
.EXAMPLE
    .\show_project_structure.ps1 -ExcludeFiles -OutputFile structure.txt
    แสดงเฉพาะโฟลเดอร์และบันทึกผลลัพธ์ลงในไฟล์ structure.txt
#>

param (
    [int]$Depth = 4,
    [switch]$ExcludeFiles,
    [string[]]$ExcludePattern = @("__pycache__", "*.pyc", "*.pyo", "*.pyd", ".git", ".venv", "venv", "env", ".ipynb_checkpoints"),
    [string]$OutputFile
)

function Get-FileIcon {
    param ([string]$Extension)
    
    switch -Wildcard ($Extension.ToLower()) {
        ".py"   { return "🐍" } # Python
        ".md"   { return "📄" } # Markdown
        ".txt"  { return "📝" } # Text
        ".json" { return "⚙️" } # JSON
        ".yaml" { return "⚙️" } # YAML
        ".yml"  { return "⚙️" } # YAML
        ".bat"  { return "🔧" } # Batch
        ".sh"   { return "🔧" } # Shell
        ".ps1"  { return "🔧" } # PowerShell
        ".jpg"  { return "🖼️" } # Image
        ".png"  { return "🖼️" } # Image
        ".gif"  { return "🖼️" } # Image
        ".jpeg" { return "🖼️" } # Image
        ".h5"   { return "🧠" } # AI model
        ".tflite" { return "🧠" } # AI model
        ".pb"   { return "🧠" } # AI model
        default { return "📎" } # Other
    }
}

function Show-DirectoryTree {
    param (
        [string]$Path,
        [string]$Indent = "",
        [string]$LastChild = "",
        [int]$CurrentDepth = 0,
        [int]$MaxDepth = 4,
        [bool]$ExcludeFiles = $false,
        [string[]]$ExcludePattern = @()
    )
    
    if ($CurrentDepth -gt $MaxDepth) { return }
    
    # Get all items, filter out excluded patterns
    $items = Get-ChildItem -Path $Path | Where-Object {
        $item = $_
        $excluded = $false
        foreach ($pattern in $ExcludePattern) {
            if ($item.Name -like $pattern) {
                $excluded = $true
                break
            }
        }
        -not $excluded
    }
    
    # Sort directories first, then files
    $directories = $items | Where-Object { $_.PSIsContainer } | Sort-Object Name
    $files = $items | Where-Object { -not $_.PSIsContainer } | Sort-Object Name
    
    # Process directories
    $directoryCount = $directories.Count
    $i = 0
    
    foreach ($directory in $directories) {
        $i++
        $isLast = ($i -eq $directoryCount) -and ($ExcludeFiles -or $files.Count -eq 0)
        
        $connector = if ($isLast) { "└── " } else { "├── " }
        $nextIndent = if ($isLast) { "    " } else { "│   " }
        
        # Output directory
        "$($Indent)$($connector)📁 $($directory.Name)"
        
        # Recursively process subdirectories
        Show-DirectoryTree -Path $directory.FullName -Indent "$($Indent)$($nextIndent)" -CurrentDepth ($CurrentDepth + 1) -MaxDepth $MaxDepth -ExcludeFiles $ExcludeFiles -ExcludePattern $ExcludePattern
    }
    
    # Process files if not excluded
    if (-not $ExcludeFiles) {
        $fileCount = $files.Count
        $i = 0
        
        foreach ($file in $files) {
            $i++
            $isLast = ($i -eq $fileCount)
            
            $connector = if ($isLast) { "└── " } else { "├── " }
            $icon = Get-FileIcon -Extension $file.Extension
            
            # Output file
            "$($Indent)$($connector)$icon $($file.Name)"
        }
    }
}

Write-Host "`n========================= Amulet-AI Project Structure =========================" -ForegroundColor Cyan
Write-Host "  MaxDepth: $Depth | ExcludeFiles: $ExcludeFiles" -ForegroundColor DarkGray
Write-Host "  Excludes: $($ExcludePattern -join ', ')" -ForegroundColor DarkGray
Write-Host "=============================================================================" -ForegroundColor Cyan

$result = @()
$result += ""
$result += "📁 Amulet-AI"
$result += (Show-DirectoryTree -Path (Get-Location) -MaxDepth $Depth -ExcludeFiles $ExcludeFiles -ExcludePattern $ExcludePattern)

$result | Where-Object { $_ -ne $null } | ForEach-Object { Write-Host $_ }

if ($OutputFile) {
    $result | Where-Object { $_ -ne $null } | Out-File -FilePath $OutputFile -Encoding utf8
    Write-Host "`nProject structure saved to: $OutputFile" -ForegroundColor Green
}

Write-Host "`n=============================================================================" -ForegroundColor Cyan
