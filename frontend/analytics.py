"""
ระบบการวัดผลและการวิเคราะห์สำหรับ Amulet-AI
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from .config import LOGGING_SETTINGS, get_absolute_path

class PerformanceMetrics:
    """คลาสสำหรับเก็บและวิเคราะห์ metrics ของระบบ"""
    
    def __init__(self):
        self.metrics = {
            "api_calls": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "avg_response_time": 0.0,
                "response_times": []
            },
            "predictions": {
                "total": 0,
                "by_class": {},
                "confidence_distribution": {
                    "high": 0,  # > 0.8
                    "medium": 0,  # 0.6-0.8
                    "low": 0  # < 0.6
                }
            },
            "user_interactions": {
                "image_uploads": 0,
                "page_views": 0,
                "session_duration": [],
                "feature_usage": {}
            },
            "system_performance": {
                "memory_usage": [],
                "cpu_usage": [],
                "disk_usage": []
            }
        }
        self.start_time = time.time()
        self.session_start = datetime.now()
        
        # ตั้งค่า logging
        self.logger = logging.getLogger(__name__)
    
    def record_api_call(self, success: bool, response_time: float, endpoint: str = "predict"):
        """บันทึกการเรียก API"""
        self.metrics["api_calls"]["total"] += 1
        self.metrics["api_calls"]["response_times"].append(response_time)
        
        if success:
            self.metrics["api_calls"]["successful"] += 1
        else:
            self.metrics["api_calls"]["failed"] += 1
        
        # คำนวณเวลาตอบสนองเฉลี่ย
        times = self.metrics["api_calls"]["response_times"]
        self.metrics["api_calls"]["avg_response_time"] = sum(times) / len(times)
        
        self.logger.info(f"API call recorded: {endpoint}, success: {success}, time: {response_time:.2f}s")
    
    def record_prediction(self, class_name: str, confidence: float):
        """บันทึกผลการทำนาย"""
        self.metrics["predictions"]["total"] += 1
        
        # นับตามคลาส
        if class_name not in self.metrics["predictions"]["by_class"]:
            self.metrics["predictions"]["by_class"][class_name] = 0
        self.metrics["predictions"]["by_class"][class_name] += 1
        
        # จัดกลุ่มตามความเชื่อมั่น
        if confidence > 0.8:
            self.metrics["predictions"]["confidence_distribution"]["high"] += 1
        elif confidence > 0.6:
            self.metrics["predictions"]["confidence_distribution"]["medium"] += 1
        else:
            self.metrics["predictions"]["confidence_distribution"]["low"] += 1
        
        self.logger.info(f"Prediction recorded: {class_name}, confidence: {confidence:.2f}")
    
    def record_user_interaction(self, action: str, details: Dict[str, Any] = None):
        """บันทึกการโต้ตอบของผู้ใช้"""
        if action == "image_upload":
            self.metrics["user_interactions"]["image_uploads"] += 1
        elif action == "page_view":
            self.metrics["user_interactions"]["page_views"] += 1
        elif action == "feature_usage":
            feature = details.get("feature", "unknown") if details else "unknown"
            if feature not in self.metrics["user_interactions"]["feature_usage"]:
                self.metrics["user_interactions"]["feature_usage"][feature] = 0
            self.metrics["user_interactions"]["feature_usage"][feature] += 1
        
        self.logger.debug(f"User interaction recorded: {action}, details: {details}")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """สร้างรายงานสรุป"""
        uptime = time.time() - self.start_time
        total_api_calls = self.metrics["api_calls"]["total"]
        success_rate = 0
        if total_api_calls > 0:
            success_rate = self.metrics["api_calls"]["successful"] / total_api_calls * 100
        
        return {
            "system_info": {
                "uptime_seconds": uptime,
                "uptime_formatted": self._format_duration(uptime),
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "api_performance": {
                "total_calls": total_api_calls,
                "success_rate": f"{success_rate:.1f}%",
                "avg_response_time": f"{self.metrics['api_calls']['avg_response_time']:.2f}s",
                "failed_calls": self.metrics["api_calls"]["failed"]
            },
            "prediction_statistics": {
                "total_predictions": self.metrics["predictions"]["total"],
                "most_predicted_classes": self._get_top_classes(5),
                "confidence_distribution": self.metrics["predictions"]["confidence_distribution"]
            },
            "user_engagement": {
                "image_uploads": self.metrics["user_interactions"]["image_uploads"],
                "page_views": self.metrics["user_interactions"]["page_views"],
                "popular_features": self._get_popular_features(3)
            }
        }
    
    def _get_top_classes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """ดึงคลาสที่ถูกทำนายมากที่สุด"""
        by_class = self.metrics["predictions"]["by_class"]
        sorted_classes = sorted(by_class.items(), key=lambda x: x[1], reverse=True)
        return [{"class": cls, "count": count} for cls, count in sorted_classes[:limit]]
    
    def _get_popular_features(self, limit: int = 3) -> List[Dict[str, Any]]:
        """ดึงฟีเจอร์ที่ใช้งานมากที่สุด"""
        features = self.metrics["user_interactions"]["feature_usage"]
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        return [{"feature": feat, "usage_count": count} for feat, count in sorted_features[:limit]]
    
    def _format_duration(self, seconds: float) -> str:
        """แปลงวินาทีเป็นรูปแบบที่อ่านง่าย"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def save_metrics(self, filepath: str = None):
        """บันทึก metrics ลงไฟล์"""
        if filepath is None:
            filepath = get_absolute_path("logs/metrics.json")
        
        # สร้างไดเรกทอรีถ้าไม่มี
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "summary": self.get_summary_report()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def load_metrics(self, filepath: str = None):
        """โหลด metrics จากไฟล์"""
        if filepath is None:
            filepath = get_absolute_path("logs/metrics.json")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metrics = data.get("metrics", self.metrics)
            self.logger.info(f"Metrics loaded from {filepath}")
        except FileNotFoundError:
            self.logger.info("No existing metrics file found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")

# สร้าง instance global สำหรับใช้งาน
performance_metrics = PerformanceMetrics()

# ฟังก์ชันสำหรับใช้งานง่าย
def record_api_call(success: bool, response_time: float, endpoint: str = "predict"):
    """บันทึกการเรียก API"""
    performance_metrics.record_api_call(success, response_time, endpoint)

def record_prediction(class_name: str, confidence: float):
    """บันทึกผลการทำนาย"""
    performance_metrics.record_prediction(class_name, confidence)

def record_user_interaction(action: str, details: Dict[str, Any] = None):
    """บันทึกการโต้ตอบของผู้ใช้"""
    performance_metrics.record_user_interaction(action, details)

def get_performance_summary():
    """รับสรุปผลการทำงาน"""
    return performance_metrics.get_summary_report()

def save_performance_data():
    """บันทึกข้อมูลประสิทธิภาพ"""
    performance_metrics.save_metrics()

def load_performance_data():
    """โหลดข้อมูลประสิทธิภาพ"""
    performance_metrics.load_metrics()
