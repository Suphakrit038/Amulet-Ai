#!/usr/bin/env python3
"""
🔄 MLOps Versioning & Tracking System
====================================

Model versioning, experiment tracking, and lifecycle management for Amulet-AI.

Features:
- Model versioning with semantic versioning
- Experiment tracking and comparison
- Model registry and artifact management
- Performance monitoring and drift detection
- Automated model deployment pipeline

Author: Amulet-AI Team
Date: October 2, 2025
"""

import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

import numpy as np
import torch
import pandas as pd
from packaging import version

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata"""
    version: str
    name: str
    architecture: str
    timestamp: datetime
    metrics: Dict[str, float]
    config: Dict[str, Any]
    artifacts_path: str
    parent_version: Optional[str] = None
    tags: List[str] = None
    notes: str = ""
    checksum: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class Experiment:
    """Experiment metadata"""
    experiment_id: str
    name: str
    description: str
    timestamp: datetime
    config: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    status: str  # 'running', 'completed', 'failed'
    duration: Optional[float] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelRegistry:
    """
    🏛️ Model Registry & Versioning System
    
    Manages model versions, tracks experiments, and handles model lifecycle.
    
    Features:
    - Semantic versioning (major.minor.patch)
    - Model comparison and rollback
    - Artifact management
    - Metadata tracking
    - Performance monitoring
    
    Example:
        >>> registry = ModelRegistry('model_registry')
        >>> version = registry.register_model(
        ...     model_path='best_model.pth',
        ...     name='amulet_classifier_v1',
        ...     metrics={'accuracy': 0.95, 'f1': 0.93}
        ... )
        >>> print(f"Registered version: {version}")
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Registry structure
        self.models_dir = self.registry_path / "models"
        self.experiments_dir = self.registry_path / "experiments"
        self.metadata_dir = self.registry_path / "metadata"
        
        for dir_path in [self.models_dir, self.experiments_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load existing registry
        self.models = self._load_models_index()
        self.experiments = self._load_experiments_index()
        
        logger.info(f"ModelRegistry initialized at {self.registry_path}")
        logger.info(f"Found {len(self.models)} models and {len(self.experiments)} experiments")
    
    def _load_models_index(self) -> Dict[str, ModelVersion]:
        """Load models index from metadata"""
        index_file = self.metadata_dir / "models_index.json"
        
        if not index_file.exists():
            return {}
        
        try:
            with open(index_file, 'r') as f:
                data = json.load(f)
            
            models = {}
            for version_str, model_data in data.items():
                # Convert timestamp string back to datetime
                model_data['timestamp'] = datetime.fromisoformat(model_data['timestamp'])
                models[version_str] = ModelVersion(**model_data)
            
            return models
            
        except Exception as e:
            logger.warning(f"Failed to load models index: {e}")
            return {}
    
    def _save_models_index(self):
        """Save models index to metadata"""
        index_file = self.metadata_dir / "models_index.json"
        
        try:
            data = {}
            for version_str, model_version in self.models.items():
                model_dict = asdict(model_version)
                # Convert datetime to string for JSON serialization
                model_dict['timestamp'] = model_version.timestamp.isoformat()
                data[version_str] = model_dict
            
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save models index: {e}")
    
    def _load_experiments_index(self) -> Dict[str, Experiment]:
        """Load experiments index from metadata"""
        index_file = self.metadata_dir / "experiments_index.json"
        
        if not index_file.exists():
            return {}
        
        try:
            with open(index_file, 'r') as f:
                data = json.load(f)
            
            experiments = {}
            for exp_id, exp_data in data.items():
                # Convert timestamp string back to datetime
                exp_data['timestamp'] = datetime.fromisoformat(exp_data['timestamp'])
                experiments[exp_id] = Experiment(**exp_data)
            
            return experiments
            
        except Exception as e:
            logger.warning(f"Failed to load experiments index: {e}")
            return {}
    
    def _save_experiments_index(self):
        """Save experiments index to metadata"""
        index_file = self.metadata_dir / "experiments_index.json"
        
        try:
            data = {}
            for exp_id, experiment in self.experiments.items():
                exp_dict = asdict(experiment)
                # Convert datetime to string for JSON serialization
                exp_dict['timestamp'] = experiment.timestamp.isoformat()
                data[exp_id] = exp_dict
            
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save experiments index: {e}")
    
    def _compute_file_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _generate_next_version(self, name: str, version_type: str = 'patch') -> str:
        """Generate next semantic version"""
        # Find existing versions for this model name
        existing_versions = [
            v.version for v in self.models.values() 
            if v.name == name
        ]
        
        if not existing_versions:
            return "1.0.0"
        
        # Parse versions and find the latest
        parsed_versions = []
        for v in existing_versions:
            try:
                parsed_versions.append(version.parse(v))
            except:
                continue
        
        if not parsed_versions:
            return "1.0.0"
        
        latest = max(parsed_versions)
        
        # Increment based on type
        if version_type == 'major':
            new_version = f"{latest.major + 1}.0.0"
        elif version_type == 'minor':
            new_version = f"{latest.major}.{latest.minor + 1}.0"
        else:  # patch
            new_version = f"{latest.major}.{latest.minor}.{latest.micro + 1}"
        
        return new_version
    
    def register_model(
        self,
        model_path: str,
        name: str,
        architecture: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        version_type: str = 'patch',
        parent_version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        auto_version: bool = True
    ) -> str:
        """
        Register a new model version
        
        Args:
            model_path: Path to model file
            name: Model name
            architecture: Model architecture name
            metrics: Performance metrics
            config: Model configuration
            version_type: 'major', 'minor', or 'patch'
            parent_version: Parent version (for tracking lineage)
            tags: Model tags
            notes: Additional notes
            auto_version: Automatically generate version number
            
        Returns:
            Version string of registered model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate version
        if auto_version:
            new_version = self._generate_next_version(name, version_type)
        else:
            # Use timestamp-based version
            new_version = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Compute checksum
        checksum = self._compute_file_checksum(str(model_path))
        
        # Create model directory
        model_dir = self.models_dir / f"{name}_{new_version}"
        model_dir.mkdir(exist_ok=True)
        
        # Copy model artifacts
        model_dest = model_dir / model_path.name
        shutil.copy2(model_path, model_dest)
        
        # Copy additional files if they exist
        model_parent = model_path.parent
        for pattern in ['*.json', '*.yaml', '*.yml', '*.txt']:
            for file_path in model_parent.glob(pattern):
                if file_path.name != model_path.name:
                    shutil.copy2(file_path, model_dir)
        
        # Create model version
        model_version = ModelVersion(
            version=new_version,
            name=name,
            architecture=architecture,
            timestamp=datetime.now(),
            metrics=metrics,
            config=config,
            artifacts_path=str(model_dir),
            parent_version=parent_version,
            tags=tags or [],
            notes=notes,
            checksum=checksum
        )
        
        # Register in index
        self.models[new_version] = model_version
        self._save_models_index()
        
        logger.info(f"Registered model {name} version {new_version}")
        logger.info(f"Artifacts saved to: {model_dir}")
        
        return new_version
    
    def get_model(self, version: str) -> Optional[ModelVersion]:
        """Get model by version"""
        return self.models.get(version)
    
    def get_latest_model(self, name: str) -> Optional[ModelVersion]:
        """Get latest version of a model by name"""
        model_versions = [
            v for v in self.models.values() 
            if v.name == name
        ]
        
        if not model_versions:
            return None
        
        # Sort by version
        try:
            sorted_versions = sorted(
                model_versions, 
                key=lambda x: version.parse(x.version),
                reverse=True
            )
            return sorted_versions[0]
        except:
            # Fallback to timestamp
            sorted_versions = sorted(
                model_versions,
                key=lambda x: x.timestamp,
                reverse=True
            )
            return sorted_versions[0]
    
    def list_models(self, name: Optional[str] = None) -> List[ModelVersion]:
        """List all models or models by name"""
        if name:
            return [v for v in self.models.values() if v.name == name]
        return list(self.models.values())
    
    def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        model1 = self.get_model(version1)
        model2 = self.get_model(version2)
        
        if not model1 or not model2:
            raise ValueError("One or both model versions not found")
        
        # Compare metrics
        metric_comparison = {}
        all_metrics = set(model1.metrics.keys()) | set(model2.metrics.keys())
        
        for metric in all_metrics:
            val1 = model1.metrics.get(metric, 0)
            val2 = model2.metrics.get(metric, 0)
            metric_comparison[metric] = {
                'model1': val1,
                'model2': val2,
                'difference': val2 - val1,
                'improvement': val2 > val1
            }
        
        return {
            'model1': asdict(model1),
            'model2': asdict(model2),
            'metric_comparison': metric_comparison,
            'architecture_match': model1.architecture == model2.architecture,
            'config_changes': self._compare_configs(model1.config, model2.config)
        }
    
    def _compare_configs(self, config1: Dict, config2: Dict) -> Dict[str, Any]:
        """Compare two configurations"""
        changes = {
            'added': {},
            'removed': {},
            'modified': {}
        }
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            if key in config1 and key in config2:
                if config1[key] != config2[key]:
                    changes['modified'][key] = {
                        'old': config1[key],
                        'new': config2[key]
                    }
            elif key in config1:
                changes['removed'][key] = config1[key]
            else:
                changes['added'][key] = config2[key]
        
        return changes
    
    def create_experiment(
        self,
        name: str,
        description: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a new experiment"""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(name) % 10000:04d}"
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            timestamp=datetime.now(),
            config=config,
            metrics={},
            artifacts=[],
            status='running',
            tags=tags or []
        )
        
        self.experiments[experiment_id] = experiment
        self._save_experiments_index()
        
        # Create experiment directory
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created experiment {experiment_id}: {name}")
        
        return experiment_id
    
    def update_experiment(
        self,
        experiment_id: str,
        metrics: Optional[Dict[str, float]] = None,
        status: Optional[str] = None,
        artifacts: Optional[List[str]] = None
    ):
        """Update experiment with new metrics or status"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if metrics:
            experiment.metrics.update(metrics)
        
        if status:
            experiment.status = status
            
            if status in ['completed', 'failed']:
                start_time = experiment.timestamp
                experiment.duration = (datetime.now() - start_time).total_seconds()
        
        if artifacts:
            experiment.artifacts.extend(artifacts)
        
        self._save_experiments_index()
        
        logger.info(f"Updated experiment {experiment_id}")
    
    def get_best_models(self, metric: str, top_k: int = 5) -> List[ModelVersion]:
        """Get top-k models by metric"""
        models_with_metric = [
            model for model in self.models.values()
            if metric in model.metrics
        ]
        
        sorted_models = sorted(
            models_with_metric,
            key=lambda x: x.metrics[metric],
            reverse=True
        )
        
        return sorted_models[:top_k]
    
    def export_registry(self, export_path: str):
        """Export entire registry to a directory"""
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Copy registry structure
        shutil.copytree(self.registry_path, export_path / "registry", dirs_exist_ok=True)
        
        # Create export summary
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'models_count': len(self.models),
            'experiments_count': len(self.experiments),
            'models': [asdict(model) for model in self.models.values()],
            'experiments': [asdict(exp) for exp in self.experiments.values()]
        }
        
        with open(export_path / "registry_export.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Registry exported to: {export_path}")


class PerformanceMonitor:
    """
    📊 Model Performance Monitoring
    
    Tracks model performance over time and detects performance drift.
    """
    
    def __init__(self, monitor_path: str = "performance_monitoring"):
        self.monitor_path = Path(monitor_path)
        self.monitor_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.monitor_path / "performance_metrics.csv"
        self.alerts_file = self.monitor_path / "alerts.json"
        
        # Initialize CSV if not exists
        if not self.metrics_file.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'model_version', 'metric_name', 'metric_value',
                'dataset', 'batch_size', 'environment'
            ])
            df.to_csv(self.metrics_file, index=False)
    
    def log_performance(
        self,
        model_version: str,
        metrics: Dict[str, float],
        dataset: str = "production",
        environment: str = "production"
    ):
        """Log performance metrics"""
        timestamp = datetime.now()
        
        # Prepare data for CSV
        rows = []
        for metric_name, metric_value in metrics.items():
            rows.append({
                'timestamp': timestamp,
                'model_version': model_version,
                'metric_name': metric_name,
                'metric_value': metric_value,
                'dataset': dataset,
                'batch_size': metrics.get('batch_size', 'unknown'),
                'environment': environment
            })
        
        # Append to CSV
        df = pd.DataFrame(rows)
        df.to_csv(self.metrics_file, mode='a', header=False, index=False)
        
        logger.info(f"Logged performance for {model_version}: {metrics}")
    
    def detect_drift(
        self,
        model_version: str,
        metric_name: str,
        window_days: int = 7,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Detect performance drift"""
        if not self.metrics_file.exists():
            return {'drift_detected': False, 'reason': 'No data available'}
        
        # Load metrics
        df = pd.read_csv(self.metrics_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter for specific model and metric
        model_data = df[
            (df['model_version'] == model_version) & 
            (df['metric_name'] == metric_name)
        ].sort_values('timestamp')
        
        if len(model_data) < 2:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Compare recent performance with baseline
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_data = model_data[model_data['timestamp'] >= cutoff_date]
        baseline_data = model_data[model_data['timestamp'] < cutoff_date]
        
        if len(recent_data) == 0 or len(baseline_data) == 0:
            return {'drift_detected': False, 'reason': 'Insufficient data for comparison'}
        
        recent_mean = recent_data['metric_value'].mean()
        baseline_mean = baseline_data['metric_value'].mean()
        
        drift_magnitude = abs(recent_mean - baseline_mean) / baseline_mean
        drift_detected = drift_magnitude > threshold
        
        return {
            'drift_detected': drift_detected,
            'drift_magnitude': drift_magnitude,
            'threshold': threshold,
            'recent_mean': recent_mean,
            'baseline_mean': baseline_mean,
            'recent_samples': len(recent_data),
            'baseline_samples': len(baseline_data)
        }


if __name__ == "__main__":
    # Example usage
    print("🔄 MLOps Versioning & Tracking System")
    print("=" * 60)
    
    # Initialize registry
    registry = ModelRegistry("example_registry")
    
    # Create experiment
    exp_id = registry.create_experiment(
        name="ResNet50 Training",
        description="Training ResNet50 on Amulet dataset",
        config={'backbone': 'resnet50', 'lr': 0.001, 'batch_size': 32}
    )
    
    # Update experiment
    registry.update_experiment(
        exp_id,
        metrics={'accuracy': 0.92, 'f1': 0.90},
        status='completed'
    )
    
    print(f"✅ Example experiment created: {exp_id}")
    print("✅ MLOps system ready for use!")