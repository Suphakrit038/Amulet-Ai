"""
System Health Monitor for Amulet-AI
Real-time monitoring and alerting system
"""

import psutil
import requests
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import threading
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart


class SystemHealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.alerts = []
        self.metrics_history = []
        self.logger = self._setup_logging()
        self.monitoring = False
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load monitoring configuration"""
        default_config = {
            "monitoring_interval": 30,  # seconds
            "alert_thresholds": {
                "cpu_usage": 80,  # percentage
                "memory_usage": 85,  # percentage
                "disk_usage": 90,  # percentage
                "response_time": 10,  # seconds
                "error_rate": 5  # percentage
            },
            "endpoints": {
                "backend": "http://localhost:8001/health",
                "frontend": "http://localhost:8501/_stcore/health"
            },
            "retention_hours": 24,
            "enable_email_alerts": False,
            "email_config": {
                "smtp_server": "",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "recipients": []
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for health monitor"""
        logger = logging.getLogger('health_monitor')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = Path('logs/health_monitor.log')
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free / (1024**3)  # GB
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "usage_percent": memory_percent,
                    "available_gb": round(memory_available, 2),
                    "total_gb": round(memory.total / (1024**3), 2)
                },
                "disk": {
                    "usage_percent": disk_percent,
                    "free_gb": round(disk_free, 2),
                    "total_gb": round(disk.total / (1024**3), 2)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "processes": process_count
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def check_service_health(self, service_name: str, endpoint: str) -> Dict:
        """Check health of a specific service"""
        try:
            start_time = time.time()
            response = requests.get(endpoint, timeout=10)
            response_time = time.time() - start_time
            
            return {
                "service": service_name,
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time": round(response_time, 3),
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.exceptions.ConnectionError:
            return {
                "service": service_name,
                "status": "unreachable",
                "status_code": 0,
                "response_time": 0,
                "timestamp": datetime.now().isoformat(),
                "error": "Connection refused"
            }
        except requests.exceptions.Timeout:
            return {
                "service": service_name,
                "status": "timeout",
                "status_code": 0,
                "response_time": 10,
                "timestamp": datetime.now().isoformat(),
                "error": "Request timeout"
            }
        except Exception as e:
            return {
                "service": service_name,
                "status": "error",
                "status_code": 0,
                "response_time": 0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def check_all_services(self) -> List[Dict]:
        """Check health of all configured services"""
        service_status = []
        
        for service_name, endpoint in self.config["endpoints"].items():
            status = self.check_service_health(service_name, endpoint)
            service_status.append(status)
        
        return service_status
    
    def analyze_metrics(self, system_metrics: Dict, service_status: List[Dict]) -> List[Dict]:
        """Analyze metrics and generate alerts"""
        alerts = []
        thresholds = self.config["alert_thresholds"]
        
        # Check system metrics
        if system_metrics:
            if system_metrics["cpu"]["usage_percent"] > thresholds["cpu_usage"]:
                alerts.append({
                    "type": "cpu_high",
                    "severity": "warning",
                    "message": f"High CPU usage: {system_metrics['cpu']['usage_percent']:.1f}%",
                    "value": system_metrics["cpu"]["usage_percent"],
                    "threshold": thresholds["cpu_usage"],
                    "timestamp": datetime.now().isoformat()
                })
            
            if system_metrics["memory"]["usage_percent"] > thresholds["memory_usage"]:
                alerts.append({
                    "type": "memory_high",
                    "severity": "warning",
                    "message": f"High memory usage: {system_metrics['memory']['usage_percent']:.1f}%",
                    "value": system_metrics["memory"]["usage_percent"],
                    "threshold": thresholds["memory_usage"],
                    "timestamp": datetime.now().isoformat()
                })
            
            if system_metrics["disk"]["usage_percent"] > thresholds["disk_usage"]:
                alerts.append({
                    "type": "disk_high",
                    "severity": "critical",
                    "message": f"High disk usage: {system_metrics['disk']['usage_percent']:.1f}%",
                    "value": system_metrics["disk"]["usage_percent"],
                    "threshold": thresholds["disk_usage"],
                    "timestamp": datetime.now().isoformat()
                })
        
        # Check service status
        for service in service_status:
            if service["status"] != "healthy":
                alerts.append({
                    "type": "service_unhealthy",
                    "severity": "critical",
                    "message": f"Service {service['service']} is {service['status']}",
                    "service": service["service"],
                    "status": service["status"],
                    "timestamp": datetime.now().isoformat()
                })
            
            if service["response_time"] > thresholds["response_time"]:
                alerts.append({
                    "type": "response_slow",
                    "severity": "warning",
                    "message": f"Slow response from {service['service']}: {service['response_time']:.3f}s",
                    "service": service["service"],
                    "response_time": service["response_time"],
                    "threshold": thresholds["response_time"],
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts
    
    def send_email_alert(self, alert: Dict):
        """Send email alert (if configured)"""
        if not self.config["enable_email_alerts"]:
            return
        
        email_config = self.config["email_config"]
        if not all([email_config["smtp_server"], email_config["username"], 
                   email_config["password"], email_config["recipients"]]):
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = email_config["username"]
            msg['To'] = ", ".join(email_config["recipients"])
            msg['Subject'] = f"Amulet-AI Alert: {alert['type']}"
            
            body = f"""
            Alert Type: {alert['type']}
            Severity: {alert['severity']}
            Message: {alert['message']}
            Timestamp: {alert['timestamp']}
            
            This is an automated alert from Amulet-AI monitoring system.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent: {alert['type']}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def save_metrics(self, system_metrics: Dict, service_status: List[Dict], alerts: List[Dict]):
        """Save metrics to file"""
        try:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "system": system_metrics,
                "services": service_status,
                "alerts": alerts
            }
            
            # Add to history
            self.metrics_history.append(metrics_data)
            
            # Clean old data
            cutoff_time = datetime.now() - timedelta(hours=self.config["retention_hours"])
            self.metrics_history = [
                m for m in self.metrics_history 
                if datetime.fromisoformat(m["timestamp"]) > cutoff_time
            ]
            
            # Save to file
            metrics_file = Path('logs/health_metrics.json')
            metrics_file.parent.mkdir(exist_ok=True)
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    def generate_health_report(self) -> Dict:
        """Generate comprehensive health report"""
        if not self.metrics_history:
            return {"error": "No metrics data available"}
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate averages over last hour
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > hour_ago
        ]
        
        if recent_metrics:
            avg_cpu = sum(m["system"]["cpu"]["usage_percent"] for m in recent_metrics if m["system"]) / len(recent_metrics)
            avg_memory = sum(m["system"]["memory"]["usage_percent"] for m in recent_metrics if m["system"]) / len(recent_metrics)
            
            # Count alerts by type
            alert_counts = {}
            for metrics in recent_metrics:
                for alert in metrics.get("alerts", []):
                    alert_type = alert["type"]
                    alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        else:
            avg_cpu = 0
            avg_memory = 0
            alert_counts = {}
        
        return {
            "report_time": datetime.now().isoformat(),
            "current_status": {
                "system": latest_metrics.get("system", {}),
                "services": latest_metrics.get("services", []),
                "active_alerts": latest_metrics.get("alerts", [])
            },
            "hourly_averages": {
                "cpu_usage": round(avg_cpu, 2),
                "memory_usage": round(avg_memory, 2)
            },
            "alert_summary": alert_counts,
            "data_points": len(self.metrics_history),
            "monitoring_uptime": "Active" if self.monitoring else "Inactive"
        }
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.logger.info("🔍 Health monitoring started")
        self.monitoring = True
        
        try:
            while self.monitoring:
                # Collect metrics
                system_metrics = self.get_system_metrics()
                service_status = self.check_all_services()
                
                # Analyze and generate alerts
                new_alerts = self.analyze_metrics(system_metrics, service_status)
                
                # Process alerts
                for alert in new_alerts:
                    self.logger.warning(f"ALERT: {alert['message']}")
                    self.send_email_alert(alert)
                
                # Save metrics
                self.save_metrics(system_metrics, service_status, new_alerts)
                
                # Wait for next interval
                time.sleep(self.config["monitoring_interval"])
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Monitoring error: {e}")
        finally:
            self.monitoring = False
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        if self.monitoring:
            self.logger.warning("Monitoring is already running")
            return
        
        monitor_thread = threading.Thread(target=self.monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return monitor_thread
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        self.logger.info("🛑 Monitoring stopped")


def main():
    """Main function for standalone monitoring"""
    monitor = SystemHealthMonitor()
    
    try:
        monitor.monitor_loop()
    except KeyboardInterrupt:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()
