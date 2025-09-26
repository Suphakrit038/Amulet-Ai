#!/usr/bin/env python3
"""
🏭 CI/CD Metrics Regression Checker
ตรวจสอบ metrics regression เพื่อป้องกัน model performance ลดลง

Usage: python ci/check_metrics_regression.py --current metrics.json --baseline baseline.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricsRegression:
    """Check for metrics regression between model versions"""
    
    def __init__(self, max_f1_drop: float = 0.02, max_accuracy_drop: float = 0.02):
        self.max_f1_drop = max_f1_drop
        self.max_accuracy_drop = max_accuracy_drop
        self.failures = []
        self.warnings = []
    
    def load_metrics(self, metrics_path: str) -> dict:
        """Load metrics from JSON file"""
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Metrics file not found: {metrics_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {metrics_path}: {e}")
            raise
    
    def compare_overall_metrics(self, current: dict, baseline: dict) -> None:
        """Compare overall model metrics"""
        logger.info("🔍 Comparing overall metrics...")
        
        current_metrics = current.get("metrics", {})
        baseline_metrics = baseline.get("metrics", {})
        
        # Compare balanced accuracy
        current_ba = current_metrics.get("balanced_accuracy", 0)
        baseline_ba = baseline_metrics.get("balanced_accuracy", 0)
        
        ba_drop = baseline_ba - current_ba
        
        if ba_drop > self.max_accuracy_drop:
            self.failures.append({
                "metric": "balanced_accuracy",
                "current": current_ba,
                "baseline": baseline_ba,
                "drop": ba_drop,
                "threshold": self.max_accuracy_drop,
                "severity": "critical"
            })
        elif ba_drop > self.max_accuracy_drop / 2:
            self.warnings.append({
                "metric": "balanced_accuracy", 
                "current": current_ba,
                "baseline": baseline_ba,
                "drop": ba_drop,
                "severity": "warning"
            })
        
        # Compare macro F1
        current_f1 = current_metrics.get("macro_f1", 0)
        baseline_f1 = baseline_metrics.get("macro_f1", 0)
        
        f1_drop = baseline_f1 - current_f1
        
        if f1_drop > self.max_f1_drop:
            self.failures.append({
                "metric": "macro_f1",
                "current": current_f1,
                "baseline": baseline_f1,
                "drop": f1_drop,
                "threshold": self.max_f1_drop,
                "severity": "critical"
            })
        elif f1_drop > self.max_f1_drop / 2:
            self.warnings.append({
                "metric": "macro_f1",
                "current": current_f1,
                "baseline": baseline_f1,
                "drop": f1_drop,
                "severity": "warning"
            })
        
        # Compare mean confidence
        current_conf = current_metrics.get("mean_confidence", 0)
        baseline_conf = baseline_metrics.get("mean_confidence", 0)
        
        conf_drop = baseline_conf - current_conf
        
        if conf_drop > 0.1:  # 10% confidence drop is concerning
            self.failures.append({
                "metric": "mean_confidence",
                "current": current_conf,
                "baseline": baseline_conf,
                "drop": conf_drop,
                "threshold": 0.1,
                "severity": "major"
            })
    
    def compare_per_class_metrics(self, current: dict, baseline: dict) -> None:
        """Compare per-class metrics"""
        logger.info("🔍 Comparing per-class metrics...")
        
        current_classes = current.get("metrics", {}).get("per_class", {})
        baseline_classes = baseline.get("metrics", {}).get("per_class", {})
        
        # Get common classes
        common_classes = set(current_classes.keys()) & set(baseline_classes.keys())
        
        if not common_classes:
            self.warnings.append({
                "metric": "class_coverage",
                "message": "No common classes found between current and baseline metrics",
                "severity": "warning"
            })
            return
        
        for class_name in common_classes:
            current_class = current_classes[class_name]
            baseline_class = baseline_classes[class_name]
            
            # Compare F1 score
            current_f1 = current_class.get("f1_score", 0)
            baseline_f1 = baseline_class.get("f1_score", 0)
            
            f1_drop = baseline_f1 - current_f1
            
            if f1_drop > self.max_f1_drop:
                self.failures.append({
                    "metric": f"{class_name}_f1_score",
                    "current": current_f1,
                    "baseline": baseline_f1,
                    "drop": f1_drop,
                    "threshold": self.max_f1_drop,
                    "severity": "major"
                })
            elif f1_drop > self.max_f1_drop / 2:
                self.warnings.append({
                    "metric": f"{class_name}_f1_score",
                    "current": current_f1,
                    "baseline": baseline_f1,
                    "drop": f1_drop,
                    "severity": "warning"
                })
            
            # Compare precision and recall
            for sub_metric in ["precision", "recall"]:
                current_value = current_class.get(sub_metric, 0)
                baseline_value = baseline_class.get(sub_metric, 0)
                
                drop = baseline_value - current_value
                
                if drop > self.max_f1_drop:
                    self.warnings.append({
                        "metric": f"{class_name}_{sub_metric}",
                        "current": current_value,
                        "baseline": baseline_value,
                        "drop": drop,
                        "severity": "warning"
                    })
    
    def compare_data_metrics(self, current: dict, baseline: dict) -> None:
        """Compare data-related metrics"""
        logger.info("🔍 Comparing data metrics...")
        
        current_total = current.get("total_samples", 0)
        baseline_total = baseline.get("total_samples", 0)
        
        # Check for significant data changes
        if current_total != baseline_total:
            sample_change = current_total - baseline_total
            percent_change = abs(sample_change) / baseline_total if baseline_total > 0 else 0
            
            if percent_change > 0.1:  # More than 10% change
                self.warnings.append({
                    "metric": "total_samples",
                    "current": current_total,
                    "baseline": baseline_total,
                    "change": sample_change,
                    "percent_change": percent_change,
                    "message": f"Sample count changed by {percent_change:.1%}",
                    "severity": "info"
                })
        
        # Compare class distribution
        current_dist = current.get("class_distribution", {})
        baseline_dist = baseline.get("class_distribution", {})
        
        if current_dist and baseline_dist:
            for class_name in set(current_dist.keys()) | set(baseline_dist.keys()):
                current_count = current_dist.get(class_name, 0)
                baseline_count = baseline_dist.get(class_name, 0)
                
                if baseline_count > 0:
                    change_percent = abs(current_count - baseline_count) / baseline_count
                    
                    if change_percent > 0.2:  # More than 20% change
                        self.warnings.append({
                            "metric": f"{class_name}_sample_count",
                            "current": current_count,
                            "baseline": baseline_count,
                            "percent_change": change_percent,
                            "message": f"Class {class_name} sample count changed by {change_percent:.1%}",
                            "severity": "info"
                        })
    
    def check_for_perfect_metrics(self, current: dict) -> None:
        """Check for suspiciously perfect metrics (potential overfitting)"""
        logger.info("🔍 Checking for perfect metrics...")
        
        current_metrics = current.get("metrics", {})
        
        # Check for perfect scores
        perfect_metrics = []
        
        if current_metrics.get("balanced_accuracy", 0) >= 0.999:
            perfect_metrics.append("balanced_accuracy")
        
        if current_metrics.get("macro_f1", 0) >= 0.999:
            perfect_metrics.append("macro_f1")
        
        # Check per-class metrics
        per_class = current_metrics.get("per_class", {})
        perfect_classes = []
        
        for class_name, class_metrics in per_class.items():
            if (class_metrics.get("f1_score", 0) >= 0.999 and 
                class_metrics.get("precision", 0) >= 0.999 and 
                class_metrics.get("recall", 0) >= 0.999):
                perfect_classes.append(class_name)
        
        if perfect_metrics or perfect_classes:
            self.warnings.append({
                "metric": "perfect_scores",
                "perfect_overall": perfect_metrics,
                "perfect_classes": perfect_classes,
                "message": "Perfect metrics detected - possible overfitting or data leakage",
                "severity": "warning",
                "recommendation": "Review data splits and consider cross-validation"
            })
    
    def run_comparison(self, current_path: str, baseline_path: str) -> dict:
        """Run complete metrics comparison"""
        logger.info("🚀 Starting metrics regression check...")
        
        # Load metrics
        current_metrics = self.load_metrics(current_path)
        baseline_metrics = self.load_metrics(baseline_path)
        
        logger.info(f"Current metrics from: {current_path}")
        logger.info(f"Baseline metrics from: {baseline_path}")
        
        # Run comparisons
        self.compare_overall_metrics(current_metrics, baseline_metrics)
        self.compare_per_class_metrics(current_metrics, baseline_metrics)
        self.compare_data_metrics(current_metrics, baseline_metrics)
        self.check_for_perfect_metrics(current_metrics)
        
        # Generate summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "current_metrics_file": current_path,
            "baseline_metrics_file": baseline_path,
            "comparison_config": {
                "max_f1_drop": self.max_f1_drop,
                "max_accuracy_drop": self.max_accuracy_drop
            },
            "results": {
                "total_failures": len(self.failures),
                "total_warnings": len(self.warnings),
                "failures": self.failures,
                "warnings": self.warnings
            },
            "overall_status": "PASS" if len(self.failures) == 0 else "FAIL"
        }
        
        logger.info("✅ Metrics regression check completed")
        
        return summary
    
    def print_summary(self, summary: dict) -> None:
        """Print comparison summary"""
        print(f"\n{'='*60}")
        print("📊 METRICS REGRESSION CHECK SUMMARY")
        print(f"{'='*60}")
        
        status = summary["overall_status"]
        status_emoji = "✅" if status == "PASS" else "❌"
        
        print(f"Status: {status_emoji} {status}")
        print(f"Failures: {summary['results']['total_failures']}")
        print(f"Warnings: {summary['results']['total_warnings']}")
        
        # Print failures
        if summary["results"]["failures"]:
            print(f"\n🚨 FAILURES ({len(summary['results']['failures'])}):")
            for failure in summary["results"]["failures"]:
                metric = failure["metric"]
                current = failure["current"]
                baseline = failure["baseline"]
                drop = failure["drop"]
                threshold = failure["threshold"]
                
                print(f"   • {metric}: {current:.4f} vs {baseline:.4f} "
                      f"(drop: {drop:.4f}, threshold: {threshold:.4f})")
        
        # Print warnings
        if summary["results"]["warnings"]:
            print(f"\n⚠️  WARNINGS ({len(summary['results']['warnings'])}):")
            for warning in summary["results"]["warnings"][:5]:  # Show top 5
                if "message" in warning:
                    print(f"   • {warning['message']}")
                else:
                    metric = warning["metric"]
                    current = warning["current"]
                    baseline = warning["baseline"]
                    drop = warning["drop"]
                    print(f"   • {metric}: {current:.4f} vs {baseline:.4f} (drop: {drop:.4f})")
        
        print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Check for metrics regression between model versions")
    parser.add_argument("--current", required=True, help="Path to current metrics JSON file")
    parser.add_argument("--baseline", required=True, help="Path to baseline metrics JSON file")
    parser.add_argument("--max-f1-drop", type=float, default=0.02, help="Maximum allowed F1 drop (default: 0.02)")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02, help="Maximum allowed accuracy drop")
    parser.add_argument("--output", help="Path to save comparison results")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit with error code if regression detected")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.current).exists():
        logger.error(f"Current metrics file not found: {args.current}")
        return 1
    
    if not Path(args.baseline).exists():
        logger.error(f"Baseline metrics file not found: {args.baseline}")
        return 1
    
    try:
        # Run regression check
        checker = MetricsRegression(
            max_f1_drop=args.max_f1_drop,
            max_accuracy_drop=args.max_accuracy_drop
        )
        
        summary = checker.run_comparison(args.current, args.baseline)
        
        # Print summary
        checker.print_summary(summary)
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Results saved to {output_path}")
        
        # Determine exit code
        exit_code = 0
        
        if summary["results"]["total_failures"] > 0:
            if args.fail_on_regression:
                logger.error("❌ Failing due to metrics regression")
                exit_code = 1
            else:
                logger.warning("⚠️ Metrics regression detected but not failing")
        
        # Special handling for perfect metrics warning
        perfect_warnings = [w for w in summary["results"]["warnings"] 
                           if w.get("metric") == "perfect_scores"]
        if perfect_warnings:
            logger.warning("⚠️ Perfect metrics detected - review for overfitting")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"💥 Regression check failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())