"""
Configuration Manager
จัดการการกำหนดค่าต่างๆ ของระบบ
"""
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """คลาสสำหรับจัดการการกำหนดค่า"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self._config = {}
        self.load_config()
    
    def load_config(self):
        """โหลดการกำหนดค่าจากไฟล์"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                self._config = self._get_default_config()
        else:
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """การกำหนดค่าเริ่มต้น"""
        return {
            "model": {
                "simulation_mode": True,
                "confidence_threshold": 0.7,
                "max_image_size": 10485760,  # 10MB
                "supported_formats": ["JPEG", "PNG", "HEIC", "WEBP", "BMP", "TIFF"]
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": True,
                "max_request_size": 10485760
            },
            "paths": {
                "dataset": "./dataset",
                "models": "./models",
                "logs": "./logs",
                "temp": "./temp"
            }
        }
    
    def get(self, key: str, default=None):
        """ดึงค่าจากการกำหนดค่า"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """ตั้งค่าใหม่"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self):
        """บันทึกการกำหนดค่า"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")

# Global config instance
config = Config()

def get_config(key: str, default=None):
    """ฟังก์ชันช่วยในการดึงค่าการกำหนดค่า"""
    return config.get(key, default)

def set_config(key: str, value: Any):
    """ฟังก์ชันช่วยในการตั้งค่า"""
    config.set(key, value)
    config.save_config()
