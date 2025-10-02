#!/usr/bin/env python3
"""
🎛️ MLOps Monitoring & Alerting System
====================================

Real-time monitoring, alerting, and dashboard for production models.

Features:
- Real-time performance monitoring
- Automated alerting system
- Model health checks
- Resource usage tracking
- Performance drift detection
- Incident management

Author: Amulet-AI Team
Date: October 2, 2025
"""

import logging
import json
import smtplib
import time
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
import queue
import psutil
import requests

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    title: str
    message: str
    timestamp: datetime
    model_version: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class MonitoringRule:
    """Monitoring rule definition"""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    threshold: float
    severity: str
    window_minutes: int = 5
    min_samples: int = 1
    enabled: bool = True


class AlertManager:
    """
    🚨 Alert Management System
    
    Manages alerts, notifications, and incident tracking.
    """
    
    def __init__(self, config_path: str = "monitoring_config.json"):
        self.config_path = Path(config_path)
        self.alerts: List[Alert] = []
        self.notification_channels: Dict[str, Any] = {}
        
        # Load configuration
        self.config = self._load_config()
        self._setup_notification_channels()
        
        logger.info("AlertManager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": ""
                },
                "webhook": {
                    "enabled": False,
                    "url": ""
                }
            }
            
            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    def _setup_notification_channels(self):
        """Setup notification channels"""
        if self.config['email']['enabled']:
            self.notification_channels['email'] = self._send_email_alert
        
        if self.config['slack']['enabled']:
            self.notification_channels['slack'] = self._send_slack_alert
        
        if self.config['webhook']['enabled']:
            self.notification_channels['webhook'] = self._send_webhook_alert
    
    def create_alert(
        self,
        severity: str,
        title: str,
        message: str,
        model_version: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None
    ) -> str:
        """Create a new alert"""
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts)}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            model_version=model_version,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        logger.warning(f"Alert created: {alert_id} - {title}")
        
        return alert_id
    
    def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                logger.info(f"Alert resolved: {alert_id}")
                
                # Send resolution notification if configured
                if self.config.get('notify_resolution', False):
                    self._send_resolution_notification(alert, resolution_note)
                
                break
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications through all enabled channels"""
        for channel_name, send_func in self.notification_channels.items():
            try:
                send_func(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel_name}: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        config = self.config['email']
        
        msg = MimeMultipart()
        msg['From'] = config['username']
        msg['To'] = ', '.join(config['recipients'])
        msg['Subject'] = f"Amulet-AI Alert: {alert.title}"
        
        body = f"""
        Alert Details:
        - Severity: {alert.severity.upper()}
        - Title: {alert.title}
        - Message: {alert.message}
        - Timestamp: {alert.timestamp}
        
        {f'- Model Version: {alert.model_version}' if alert.model_version else ''}
        {f'- Metric: {alert.metric_name} = {alert.metric_value}' if alert.metric_name else ''}
        {f'- Threshold: {alert.threshold}' if alert.threshold else ''}
        
        Alert ID: {alert.alert_id}
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['username'], config['password'])
        server.send_message(msg)
        server.quit()
    
    def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        webhook_url = self.config['slack']['webhook_url']
        
        color = {
            'low': 'good',
            'medium': 'warning',
            'high': 'danger',
            'critical': 'danger'
        }.get(alert.severity, 'warning')
        
        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"Amulet-AI Alert: {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.upper(), "short": True},
                        {"title": "Alert ID", "value": alert.alert_id, "short": True},
                        {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": True}
                    ]
                }
            ]
        }
        
        if alert.model_version:
            payload["attachments"][0]["fields"].append({
                "title": "Model Version", "value": alert.model_version, "short": True
            })
        
        if alert.metric_name:
            payload["attachments"][0]["fields"].append({
                "title": "Metric", "value": f"{alert.metric_name} = {alert.metric_value}", "short": True
            })
        
        requests.post(webhook_url, json=payload)
    
    def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        webhook_url = self.config['webhook']['url']
        
        payload = {
            "alert_type": "model_monitoring",
            "alert": asdict(alert)
        }
        
        requests.post(webhook_url, json=payload)
    
    def _send_resolution_notification(self, alert: Alert, resolution_note: str):
        """Send alert resolution notification"""
        resolution_alert = Alert(
            alert_id=f"resolved_{alert.alert_id}",
            severity="low",
            title=f"RESOLVED: {alert.title}",
            message=f"Alert {alert.alert_id} has been resolved. {resolution_note}",
            timestamp=datetime.now()
        )
        
        self._send_notifications(resolution_alert)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """Get alerts by severity level"""
        return [alert for alert in self.alerts if alert.severity == severity]


class ModelHealthChecker:
    """
    🏥 Model Health Monitoring
    
    Performs health checks on deployed models.
    """
    
    def __init__(self, model_endpoint: str = "http://localhost:8000"):
        self.model_endpoint = model_endpoint
        self.health_history: List[Dict[str, Any]] = []
    
    def check_model_health(self) -> Dict[str, Any]:
        """Perform comprehensive model health check"""
        health_status = {
            'timestamp': datetime.now(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # API availability check
        health_status['checks']['api_availability'] = self._check_api_availability()
        
        # Response time check
        health_status['checks']['response_time'] = self._check_response_time()
        
        # Model inference check
        health_status['checks']['inference'] = self._check_inference()
        
        # Resource usage check
        health_status['checks']['resources'] = self._check_resource_usage()
        
        # Determine overall status
        if any(check['status'] == 'critical' for check in health_status['checks'].values()):
            health_status['overall_status'] = 'critical'
        elif any(check['status'] == 'warning' for check in health_status['checks'].values()):
            health_status['overall_status'] = 'warning'
        
        self.health_history.append(health_status)
        
        return health_status
    
    def _check_api_availability(self) -> Dict[str, Any]:
        """Check if API is available"""
        try:
            response = requests.get(f"{self.model_endpoint}/health", timeout=10)
            if response.status_code == 200:
                return {'status': 'healthy', 'message': 'API is available'}
            else:
                return {'status': 'warning', 'message': f'API returned status {response.status_code}'}
        except Exception as e:
            return {'status': 'critical', 'message': f'API unavailable: {str(e)}'}
    
    def _check_response_time(self) -> Dict[str, Any]:
        """Check API response time"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.model_endpoint}/health", timeout=10)
            response_time = time.time() - start_time
            
            if response_time < 1.0:
                status = 'healthy'
            elif response_time < 3.0:
                status = 'warning'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'response_time': response_time,
                'message': f'Response time: {response_time:.2f}s'
            }
        except Exception as e:
            return {'status': 'critical', 'message': f'Response time check failed: {str(e)}'}
    
    def _check_inference(self) -> Dict[str, Any]:
        """Check model inference capability"""
        try:
            # Create dummy image data for testing
            test_data = {
                'image': 'dummy_base64_image_data',
                'test': True
            }
            
            response = requests.post(
                f"{self.model_endpoint}/predict",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'prediction' in result or 'test_response' in result:
                    return {'status': 'healthy', 'message': 'Inference working correctly'}
                else:
                    return {'status': 'warning', 'message': 'Unexpected inference response format'}
            else:
                return {'status': 'warning', 'message': f'Inference returned status {response.status_code}'}
                
        except Exception as e:
            return {'status': 'critical', 'message': f'Inference check failed: {str(e)}'}
    
    def _check_resource_usage(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = 'healthy'
            issues = []
            
            if cpu_percent > 90:
                status = 'critical'
                issues.append(f'High CPU usage: {cpu_percent:.1f}%')
            elif cpu_percent > 70:
                status = 'warning'
                issues.append(f'Elevated CPU usage: {cpu_percent:.1f}%')
            
            if memory.percent > 90:
                status = 'critical'
                issues.append(f'High memory usage: {memory.percent:.1f}%')
            elif memory.percent > 70:
                status = 'warning'
                issues.append(f'Elevated memory usage: {memory.percent:.1f}%')
            
            if disk.percent > 90:
                status = 'critical'
                issues.append(f'High disk usage: {disk.percent:.1f}%')
            elif disk.percent > 80:
                status = 'warning'
                issues.append(f'Elevated disk usage: {disk.percent:.1f}%')
            
            message = '; '.join(issues) if issues else 'Resource usage normal'
            
            return {
                'status': status,
                'message': message,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent
            }
            
        except Exception as e:
            return {'status': 'warning', 'message': f'Resource check failed: {str(e)}'}


class RealTimeMonitor:
    """
    📡 Real-time Model Monitoring
    
    Continuously monitors model performance and triggers alerts.
    """
    
    def __init__(self, alert_manager: AlertManager, health_checker: ModelHealthChecker):
        self.alert_manager = alert_manager
        self.health_checker = health_checker
        self.monitoring_rules: List[MonitoringRule] = []
        self.metrics_queue = queue.Queue()
        self.running = False
        self.monitor_thread = None
        
        # Load default monitoring rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default monitoring rules"""
        default_rules = [
            MonitoringRule(
                rule_id="accuracy_drop",
                name="Accuracy Drop",
                metric_name="accuracy",
                condition="lt",
                threshold=0.8,
                severity="high",
                window_minutes=10
            ),
            MonitoringRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                metric_name="error_rate",
                condition="gt",
                threshold=0.1,
                severity="critical",
                window_minutes=5
            ),
            MonitoringRule(
                rule_id="slow_response",
                name="Slow Response Time",
                metric_name="response_time",
                condition="gt",
                threshold=5.0,
                severity="medium",
                window_minutes=5
            )
        ]
        
        self.monitoring_rules.extend(default_rules)
    
    def add_monitoring_rule(self, rule: MonitoringRule):
        """Add a new monitoring rule"""
        self.monitoring_rules.append(rule)
        logger.info(f"Added monitoring rule: {rule.name}")
    
    def log_metric(self, metric_name: str, value: float, model_version: str = "current"):
        """Log a metric for monitoring"""
        metric_data = {
            'timestamp': datetime.now(),
            'metric_name': metric_name,
            'value': value,
            'model_version': model_version
        }
        
        self.metrics_queue.put(metric_data)
    
    def start_monitoring(self, check_interval: int = 60):
        """Start real-time monitoring"""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"Started real-time monitoring (check interval: {check_interval}s)")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=30)
        
        logger.info("Stopped real-time monitoring")
    
    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop"""
        while self.running:
            try:
                # Process queued metrics
                self._process_metrics()
                
                # Perform health checks
                health_status = self.health_checker.check_model_health()
                self._check_health_alerts(health_status)
                
                # Sleep until next check
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def _process_metrics(self):
        """Process metrics from the queue"""
        recent_metrics = {}
        
        # Collect all metrics from queue
        while not self.metrics_queue.empty():
            try:
                metric_data = self.metrics_queue.get_nowait()
                metric_name = metric_data['metric_name']
                
                if metric_name not in recent_metrics:
                    recent_metrics[metric_name] = []
                
                recent_metrics[metric_name].append(metric_data)
                
            except queue.Empty:
                break
        
        # Check each monitoring rule
        for rule in self.monitoring_rules:
            if not rule.enabled:
                continue
            
            if rule.metric_name in recent_metrics:
                self._check_rule(rule, recent_metrics[rule.metric_name])
    
    def _check_rule(self, rule: MonitoringRule, metric_data: List[Dict]):
        """Check if a monitoring rule is violated"""
        if len(metric_data) < rule.min_samples:
            return
        
        # Filter data within time window
        cutoff_time = datetime.now() - timedelta(minutes=rule.window_minutes)
        recent_data = [
            data for data in metric_data 
            if data['timestamp'] >= cutoff_time
        ]
        
        if len(recent_data) < rule.min_samples:
            return
        
        # Calculate average value
        values = [data['value'] for data in recent_data]
        avg_value = np.mean(values)
        
        # Check condition
        violation = False
        if rule.condition == 'gt' and avg_value > rule.threshold:
            violation = True
        elif rule.condition == 'lt' and avg_value < rule.threshold:
            violation = True
        elif rule.condition == 'gte' and avg_value >= rule.threshold:
            violation = True
        elif rule.condition == 'lte' and avg_value <= rule.threshold:
            violation = True
        elif rule.condition == 'eq' and abs(avg_value - rule.threshold) < 1e-6:
            violation = True
        
        if violation:
            self.alert_manager.create_alert(
                severity=rule.severity,
                title=rule.name,
                message=f"Rule '{rule.name}' violated: {rule.metric_name} = {avg_value:.4f} (threshold: {rule.threshold})",
                model_version=recent_data[0].get('model_version'),
                metric_name=rule.metric_name,
                metric_value=avg_value,
                threshold=rule.threshold
            )
    
    def _check_health_alerts(self, health_status: Dict[str, Any]):
        """Check health status and create alerts if needed"""
        if health_status['overall_status'] == 'critical':
            critical_checks = [
                name for name, check in health_status['checks'].items()
                if check['status'] == 'critical'
            ]
            
            self.alert_manager.create_alert(
                severity='critical',
                title='Model Health Critical',
                message=f"Critical health issues detected: {', '.join(critical_checks)}"
            )
        
        elif health_status['overall_status'] == 'warning':
            warning_checks = [
                name for name, check in health_status['checks'].items()
                if check['status'] == 'warning'
            ]
            
            self.alert_manager.create_alert(
                severity='medium',
                title='Model Health Warning',
                message=f"Health warnings detected: {', '.join(warning_checks)}"
            )


def setup_monitoring_system(
    model_endpoint: str = "http://localhost:8000",
    config_path: str = "monitoring_config.json"
) -> Tuple[AlertManager, ModelHealthChecker, RealTimeMonitor]:
    """
    Setup complete monitoring system
    
    Returns:
        Tuple of (AlertManager, ModelHealthChecker, RealTimeMonitor)
    """
    # Initialize components
    alert_manager = AlertManager(config_path)
    health_checker = ModelHealthChecker(model_endpoint)
    monitor = RealTimeMonitor(alert_manager, health_checker)
    
    logger.info("Monitoring system setup complete")
    
    return alert_manager, health_checker, monitor


if __name__ == "__main__":
    # Example usage
    print("🎛️ MLOps Monitoring & Alerting System")
    print("=" * 60)
    
    # Setup monitoring
    alert_manager, health_checker, monitor = setup_monitoring_system()
    
    # Start monitoring
    monitor.start_monitoring(check_interval=30)
    
    # Simulate some metrics
    monitor.log_metric('accuracy', 0.95)
    monitor.log_metric('response_time', 1.2)
    
    print("✅ Monitoring system started!")
    print("✅ Check logs for monitoring activities")
    
    # Keep running for demonstration
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("\n✅ Monitoring stopped")