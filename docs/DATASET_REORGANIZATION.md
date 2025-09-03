# Amulet-AI Dataset Reorganization

## Overview

This document describes the reorganization of the Amulet-AI dataset folders to improve organization and structure.

## Changes Made

- Renamed Thai folder names to English equivalents for better compatibility
- Organized images into front_color and back_color subdirectories
- Created a consistent folder structure across train, test, and validation sets

## New Folder Structure

```
dataset_split/
├── test/
│   ├── buddha_in_vihara/
│   │   ├── front_color/
│   │   └── back_color/
│   ├── phra_san/
│   │   ├── front_color/
│   │   └── back_color/
│   ├── phra_sivali/
│   │   ├── front_color/
│   │   └── back_color/
│   ├── somdej_fatherguay/
│   │   ├── front_color/
│   │   └── back_color/
│   ├── somdej_portrait_back/
│   │   ├── front_color/
│   │   └── back_color/
│   ├── somdej_prok_bodhi/
│   │   ├── front_color/
│   │   └── back_color/
│   ├── somdej_thansing/
│   │   ├── front_color/
│   │   └── back_color/
│   ├── somdej_waek_man/
│   │   ├── front_color/
│   │   └── back_color/
│   └── wat_nong_e_duk/
│       ├── front_color/
│       └── back_color/
├── train/
│   ├── [same structure as test]
└── validation/
    ├── [same structure as test]
```

## Folder Name Mappings

| Original Thai Name | New English Name |
|-------------------|------------------|
| พระพุทธเจ้าในวิหาร | buddha_in_vihara |
| พระสมเด็จฐานสิงห์ | somdej_thansing |
| พระสมเด็จประทานพร พุทธกวัก | somdej_pudtagueg |
| พระสมเด็จหลังรูปเหมือน | somdej_portrait_back |
| พระสรรค์ | phra_san |
| พระสิวลี | phra_sivali |
| สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ | somdej_prok_bodhi |
| สมเด็จแหวกม่าน | somdej_waek_man |
| ออกวัดหนองอีดุก | wat_nong_e_duk |

## Statistics

- Total files reorganized: 401
  - Main dataset: 170 files
  - Dataset split: 231 files
- Files with errors: 0

## Front/Back Organization

Images were sorted based on their filenames:
- Files containing "front" in the filename were placed in the front_color directory
- Files containing "back" in the filename were placed in the back_color directory

## Benefits

1. Better compatibility with various operating systems and tools
2. Consistent naming convention across the project
3. Improved organization by separating front and back views
4. Enhanced structure for machine learning pipeline processing

## Date of Reorganization

Reorganization completed on: September 3, 2023
