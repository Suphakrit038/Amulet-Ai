# 🔍 Dataset Inspection Report
**Dataset Path:** `dataset`
**Scan Date:** 2025-09-25 23:16:33

## 📊 Executive Summary

- **Total Images:** 170
- **Classes:** 3
- **Average per Class:** 56.7
- **Class Range:** 20 - 100
- **Imbalance Ratio:** 5.0:1
- **Corrupted Files:** 0
- **Duplicates:** 85

## 🎯 Cross-Validation Strategy

**Recommended K-Fold:** 5
**Strategy Note:** ✅ Use 5-fold stratified CV. Dataset size adequate for robust validation.

## 📋 Per-Class Analysis

| Class | Images | Corrupted | Duplicates | Health | Aug Factor | Sample Sizes |
|-------|---------|-----------|------------|---------|------------|--------------|
| phra_nang_phya | 50 | 0 | 25 | ✅ healthy | 0x | 224x224x3(50) |
| phra_rod | 20 | 0 | 10 | ✅ healthy | 1x | 224x224x3(20) |
| phra_somdej | 100 | 0 | 50 | ✅ healthy | 0x | 224x224x3(100) |

## 🎯 Recommendations

1. ⚠️ MANY DUPLICATES: 85 duplicate images (50.0%). Consider deduplication.

## 📸 Sample Files (First 3 per class)

**phra_nang_phya:**
- `dataset\phra_nang_phya\phra_nang_phya_000.jpg`
- `dataset\phra_nang_phya\phra_nang_phya_001.jpg`
- `dataset\phra_nang_phya\phra_nang_phya_002.jpg`

**phra_rod:**
- `dataset\phra_rod\phra_rod_000.jpg`
- `dataset\phra_rod\phra_rod_001.jpg`
- `dataset\phra_rod\phra_rod_002.jpg`

**phra_somdej:**
- `dataset\phra_somdej\phra_somdej_000.jpg`
- `dataset\phra_somdej\phra_somdej_001.jpg`
- `dataset\phra_somdej\phra_somdej_002.jpg`
