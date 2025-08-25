"""
Error Handling Utilities
ระบบจัดการข้อผิดพลาดแบบครบวงจร
"""
from typing import Dict, Any, Optional, List
from enum import Enum
import traceback
from datetime import datetime

class ErrorType(Enum):
    """ประเภทของข้อผิดพลาด"""
    VALIDATION_ERROR = "validation_error"
    MODEL_ERROR = "model_error"
    IMAGE_PROCESSING_ERROR = "image_processing_error"
    API_ERROR = "api_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"

class AmuletError(Exception):
    """Base exception class สำหรับ Amulet AI"""
    
    def __init__(self, message: str, error_type: ErrorType = ErrorType.UNKNOWN_ERROR, 
                 details: Optional[Dict] = None, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """แปลง error เป็น dictionary"""
        return {
            "message": self.message,
            "error_type": self.error_type.value,
            "details": self.details,
            "status_code": self.status_code,
            "timestamp": self.timestamp.isoformat(),
            "success": False
        }

class ValidationError(AmuletError):
    """ข้อผิดพลาดจากการ validate input"""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)
        
        super().__init__(
            message=message,
            error_type=ErrorType.VALIDATION_ERROR,
            details=details,
            status_code=400
        )

class ModelError(AmuletError):
    """ข้อผิดพลาดจาก AI model"""
    
    def __init__(self, message: str, model_name: str = None):
        details = {}
        if model_name:
            details["model_name"] = model_name
        
        super().__init__(
            message=message,
            error_type=ErrorType.MODEL_ERROR,
            details=details,
            status_code=500
        )

class ImageProcessingError(AmuletError):
    """ข้อผิดพลาดจากการประมวลผลภาพ"""
    
    def __init__(self, message: str, image_info: Dict = None):
        super().__init__(
            message=message,
            error_type=ErrorType.IMAGE_PROCESSING_ERROR,
            details=image_info or {},
            status_code=400
        )

def handle_error(func):
    """Decorator สำหรับจัดการข้อผิดพลาด"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AmuletError:
            raise  # Re-raise custom errors
        except Exception as e:
            # Convert unknown errors to AmuletError
            raise AmuletError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_type=ErrorType.SYSTEM_ERROR,
                details={"function": func.__name__, "original_error": str(e)}
            )
    return wrapper

def create_error_response(error: Exception) -> Dict[str, Any]:
    """สร้าง error response แบบมาตรฐาน"""
    if isinstance(error, AmuletError):
        return error.to_dict()
    else:
        # Handle unexpected errors
        return {
            "message": str(error),
            "error_type": ErrorType.UNKNOWN_ERROR.value,
            "details": {"original_error": str(error)},
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
            "success": False
        }

def validate_required_fields(data: Dict, required_fields: List[str]) -> None:
    """ตรวจสอบ required fields"""
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            field=missing_fields[0] if len(missing_fields) == 1 else "multiple"
        )
