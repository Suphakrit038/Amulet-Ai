# API Specification - Amulet-AI

## Overview
REST API สำหรับ Amulet-AI ระบบวิเคราะห์พระเครื่องด้วย AI

**Base URL:** `http://localhost:8000`  
**Version:** v1.0  
**Content-Type:** `application/json`  

## Authentication
ปัจจุบันไม่ต้องการ authentication สำหรับการใช้งานในเครื่อง

---

## Endpoints

### 1. Health Check

#### `GET /health`
ตรวจสอบสถานะ API server

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 3600
}
```

**Status Codes:**
- `200` - API ทำงานปกติ
- `503` - API มีปัญหา

---

### 2. Single Image Analysis

#### `POST /api/v1/analyze/single`
วิเคราะห์พระเครื่องจากรูปภาพเดียว

**Request:**
```http
POST /api/v1/analyze/single
Content-Type: multipart/form-data

image: [binary file data]
enhance: true (optional, default: false)
confidence_threshold: 0.7 (optional, default: 0.7)
```

**Parameters:**
- `image` (file, required) - ไฟล์รูปภาพ (PNG, JPG, JPEG)
- `enhance` (boolean, optional) - ปรับปรุงภาพอัตโนมัติ
- `confidence_threshold` (float, optional) - เกณฑ์ความมั่นใจ (0.0-1.0)

**Response:**
```json
{
  "success": true,
  "data": {
    "predicted_class": "somdej",
    "thai_name": "พระสมเด็จ",
    "confidence": 0.89,
    "analysis_type": "single_image",
    "processing_time": 1.42,
    "model_version": "Enhanced CNN v2.1",
    "timestamp": "2025-01-01T12:00:00Z",
    "top_predictions": [
      {
        "class": "somdej",
        "thai_name": "พระสมเด็จ",
        "confidence": 0.89
      },
      {
        "class": "nang_phaya",
        "thai_name": "พระนางพญา", 
        "confidence": 0.08
      },
      {
        "class": "pim_lek",
        "thai_name": "พระพิมพ์เล็ก",
        "confidence": 0.03
      }
    ],
    "image_info": {
      "original_size": [800, 600],
      "processed_size": [224, 224],
      "quality_score": 0.85,
      "was_enhanced": true
    }
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "ไฟล์รูปภาพไม่ถูกต้องหรือเสียหาย",
    "details": "Unable to decode image file"
  }
}
```

**Status Codes:**
- `200` - วิเคราะห์สำเร็จ
- `400` - ข้อมูลที่ส่งมาไม่ถูกต้อง
- `413` - ไฟล์ใหญ่เกินไป
- `422` - ไฟล์ไม่ใช่รูปภาพที่รองรับ
- `500` - เกิดข้อผิดพลาดภายใน

---

### 3. Dual Image Analysis

#### `POST /api/v1/analyze/dual`
วิเคราะห์พระเครื่องจากรูปภาพคู่ (หน้า-หลัง)

**Request:**
```http
POST /api/v1/analyze/dual
Content-Type: multipart/form-data

front_image: [binary file data]
back_image: [binary file data]
enhance: true (optional)
confidence_threshold: 0.7 (optional)
```

**Parameters:**
- `front_image` (file, required) - รูปด้านหน้า
- `back_image` (file, required) - รูปด้านหลัง
- `enhance` (boolean, optional) - ปรับปรุงภาพอัตโนมัติ
- `confidence_threshold` (float, optional) - เกณฑ์ความมั่นใจ

**Response:**
```json
{
  "success": true,
  "data": {
    "predicted_class": "somdej",
    "thai_name": "พระสมเด็จ",
    "confidence": 0.94,
    "analysis_type": "dual_image",
    "processing_time": 2.18,
    "model_version": "Enhanced Dual-View CNN v2.1",
    "timestamp": "2025-01-01T12:00:00Z",
    "top_predictions": [
      {
        "class": "somdej",
        "thai_name": "พระสมเด็จ",
        "confidence": 0.94
      },
      {
        "class": "nang_phaya", 
        "thai_name": "พระนางพญา",
        "confidence": 0.04
      },
      {
        "class": "pim_lek",
        "thai_name": "พระพิมพ์เล็ก",
        "confidence": 0.02
      }
    ],
    "image_info": {
      "front_image": {
        "original_size": [800, 600],
        "quality_score": 0.87,
        "was_enhanced": true
      },
      "back_image": {
        "original_size": [800, 600], 
        "quality_score": 0.82,
        "was_enhanced": true
      }
    },
    "dual_analysis_features": {
      "cross_validation_score": 0.96,
      "consistency_check": "passed",
      "confidence_boost": 0.09
    }
  }
}
```

---

### 4. Model Information

#### `GET /api/v1/model/info`
ข้อมูลเกี่ยวกับ AI model

**Response:**
```json
{
  "success": true,
  "data": {
    "model_name": "Enhanced Multilayer CNN",
    "version": "2.1.0",
    "architecture": "CNN",
    "input_size": [224, 224, 3],
    "output_classes": 10,
    "accuracy": 0.945,
    "f1_score": 0.932,
    "training_data": {
      "total_images": 75000,
      "classes": 10,
      "validation_split": 0.2
    },
    "performance": {
      "avg_inference_time": 1.2,
      "memory_usage": "512MB",
      "gpu_acceleration": true
    },
    "last_updated": "2024-12-15T10:00:00Z"
  }
}
```

---

### 5. Supported Classes

#### `GET /api/v1/classes`
รายการประเภทพระเครื่องที่รองรับ

**Response:**
```json
{
  "success": true,
  "data": {
    "total_classes": 10,
    "classes": [
      {
        "id": "somdej",
        "thai_name": "พระสมเด็จ",
        "english_name": "Phra Somdej",
        "description": "พระเครื่องที่มีชื่อเสียงและเป็นที่นิยมสูง"
      },
      {
        "id": "nang_phaya",
        "thai_name": "พระนางพญา",
        "english_name": "Phra Nang Phaya",
        "description": "พระเครื่องโบราณที่มีรูปลักษณ์เป็นหญิง"
      },
      {
        "id": "pim_lek",
        "thai_name": "พระพิมพ์เล็ก",
        "english_name": "Phra Pim Lek", 
        "description": "พระเครื่องขนาดเล็กที่นิยมใส่คอ"
      }
    ]
  }
}
```

---

### 6. Image Enhancement

#### `POST /api/v1/enhance`
ปรับปรุงคุณภาพภาพแยกต่างหาก

**Request:**
```http
POST /api/v1/enhance
Content-Type: multipart/form-data

image: [binary file data]
enhancement_level: medium (optional)
```

**Parameters:**
- `image` (file, required) - ไฟล์รูปภาพ
- `enhancement_level` (string, optional) - ระดับการปรับปรุง: `low`, `medium`, `high`

**Response:**
```json
{
  "success": true,
  "data": {
    "enhanced_image": "[base64 encoded image]",
    "enhancement_info": {
      "original_quality": 0.65,
      "enhanced_quality": 0.84,
      "enhancements_applied": [
        "brightness",
        "contrast", 
        "sharpness"
      ],
      "processing_time": 0.8
    }
  }
}
```

---

## Error Codes

| Code | Message | Description |
|------|---------|-------------|
| `INVALID_IMAGE` | Invalid image file | ไฟล์ไม่ใช่รูปภาพหรือเสียหาย |
| `FILE_TOO_LARGE` | File size exceeds limit | ไฟล์ใหญ่เกิน 10MB |
| `UNSUPPORTED_FORMAT` | Unsupported image format | รูปแบบไฟล์ไม่รองรับ |
| `MISSING_PARAMETER` | Required parameter missing | ขาดพารามิเตอร์ที่จำเป็น |
| `MODEL_ERROR` | AI model error | เกิดข้อผิดพลาดใน AI model |
| `PROCESSING_ERROR` | Image processing error | เกิดข้อผิดพลาดในการประมวลผลภาพ |
| `SERVER_ERROR` | Internal server error | เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์ |

---

## Rate Limiting
- **Development:** ไม่จำกัด
- **Production:** 100 requests/minute per IP

---

## File Constraints
- **Supported formats:** PNG, JPG, JPEG
- **Maximum file size:** 10MB
- **Minimum resolution:** 100x100 pixels
- **Maximum resolution:** 4000x4000 pixels

---

## Example Usage

### cURL
```bash
# Single image analysis
curl -X POST http://localhost:8000/api/v1/analyze/single \
  -F "image=@amulet.jpg" \
  -F "enhance=true"

# Dual image analysis  
curl -X POST http://localhost:8000/api/v1/analyze/dual \
  -F "front_image=@front.jpg" \
  -F "back_image=@back.jpg" \
  -F "enhance=true"

# Health check
curl http://localhost:8000/health
```

### Python
```python
import requests

# Single image analysis
with open('amulet.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/analyze/single',
        files={'image': f},
        data={'enhance': 'true'}
    )
    
result = response.json()
print(f"Prediction: {result['data']['thai_name']}")
print(f"Confidence: {result['data']['confidence']:.2%}")
```

### JavaScript
```javascript
// Single image analysis
const formData = new FormData();
formData.append('image', imageFile);
formData.append('enhance', 'true');

fetch('http://localhost:8000/api/v1/analyze/single', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Prediction:', data.data.thai_name);
    console.log('Confidence:', data.data.confidence);
});
```

---

## Changelog

### v1.0.0 (2025-01-01)
- เพิ่ม single image analysis
- เพิ่ม dual image analysis  
- เพิ่ม image enhancement
- เพิ่ม model information API
- รองรับ 10 ประเภทพระเครื่อง