#!/usr/bin/env python3
"""
📊 Performance Monitoring System
ระบบติดตามประสิทธิภาพแบบ real-time
"""
import time
import psutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import threading
import queue

class PerformanceMonitor:
    def __init__(self, log_interval: int = 60):
        self.log_interval = log_interval
        self.metrics_queue = queue.Queue()
        self.is_monitoring = False
        self.log_file = Path("logs") / "performance_metrics.json"
        self.log_file.parent.mkdir(exist_ok=True)
        
    def start_monitoring(self):
        """เริ่มการติดตาม"""
        if not self.is_monitoring:
            self.is_monitoring = True
            monitor_thread = threading.Thread(target=self._monitoring_loop)
            monitor_thread.daemon = True
            monitor_thread.start()
            print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """หยุดการติดตาม"""
        self.is_monitoring = False
        print("📊 Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Loop การติดตาม"""
        while self.is_monitoring:
            metrics = self.collect_metrics()
            self.metrics_queue.put(metrics)
            self._save_metrics(metrics)
            time.sleep(self.log_interval)
    
    def collect_metrics(self) -> Dict:
        """เก็บข้อมูล metrics"""
        try:
            process = psutil.Process()
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage_percent": psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
                },
                "process": {
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "num_threads": process.num_threads(),
                    "num_fds": process.num_fds() if hasattr(process, 'num_fds') else 0
                }
            }
            
            return metrics
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _save_metrics(self, metrics: Dict):
        """บันทึก metrics"""
        try:
            # อ่านข้อมูลเก่า
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {"metrics": []}
            
            # เพิ่มข้อมูลใหม่
            existing_data["metrics"].append(metrics)
            
            # เก็บแค่ 1000 records ล่าสุด
            if len(existing_data["metrics"]) > 1000:
                existing_data["metrics"] = existing_data["metrics"][-1000:]
            
            # บันทึก
            with open(self.log_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    def get_recent_metrics(self, count: int = 10) -> List[Dict]:
        """ดึงข้อมูล metrics ล่าสุด"""
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                return data["metrics"][-count:]
            return []
        except Exception as e:
            print(f"Error reading metrics: {e}")
            return []
    
    def get_performance_summary(self) -> Dict:
        """สรุปประสิทธิภาพ"""
        metrics = self.get_recent_metrics(100)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        # คำนวณค่าเฉลี่ย
        cpu_values = [m["system"]["cpu_percent"] for m in metrics if "system" in m]
        memory_values = [m["system"]["memory_percent"] for m in metrics if "system" in m]
        process_memory = [m["process"]["memory_mb"] for m in metrics if "process" in m]
        
        summary = {
            "period": f"Last {len(metrics)} records",
            "system_cpu_avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "system_memory_avg": sum(memory_values) / len(memory_values) if memory_values else 0,
            "process_memory_avg": sum(process_memory) / len(process_memory) if process_memory else 0,
            "process_memory_max": max(process_memory) if process_memory else 0,
            "last_update": metrics[-1]["timestamp"] if metrics else None
        }
        
        return summary

# สร้าง global monitor instance
performance_monitor = PerformanceMonitor()

def start_performance_monitoring():
    """เริ่มการติดตามประสิทธิภาพ"""
    performance_monitor.start_monitoring()

def stop_performance_monitoring():
    """หยุดการติดตามประสิทธิภาพ"""
    performance_monitor.stop_monitoring()

def get_performance_status():
    """ดูสถานะประสิทธิภาพปัจจุบัน"""
    return performance_monitor.get_performance_summary()
