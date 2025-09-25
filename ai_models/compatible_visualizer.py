"""
📊 Compatible Visualization Module for Python 3.13
แทนที่ matplotlib ด้วย plotly และ bokeh
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from pathlib import Path

class CompatibleVisualizer:
    """
    🎨 Visualization class ที่ทำงานได้กับ Python 3.13
    ใช้ plotly แทน matplotlib
    """
    
    def __init__(self, output_dir: str = "ai_models/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette
        self.colors = px.colors.qualitative.Set3
    
    def plot_class_distribution(self, class_counts: Dict[str, int], 
                               title: str = "Class Distribution",
                               save_path: Optional[str] = None) -> go.Figure:
        """Plot class distribution using plotly"""
        
        # Prepare data
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(
                x=classes,
                y=counts,
                marker_color=self.colors[:len(classes)],
                text=counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Classes",
            yaxis_title="Number of Samples",
            showlegend=False,
            height=500,
            xaxis_tickangle=-45
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(str(self.output_dir / f"{save_path}.html"))
            fig.write_image(str(self.output_dir / f"{save_path}.png"))
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importances: List[float],
                               title: str = "Feature Importance",
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> go.Figure:
        """Plot feature importance"""
        
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1][:top_n]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_importances,
                y=sorted_names,
                orientation='h',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Features",
            height=max(400, len(sorted_names) * 25),
        )
        
        if save_path:
            fig.write_html(str(self.output_dir / f"{save_path}.html"))
            fig.write_image(str(self.output_dir / f"{save_path}.png"))
        
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                             class_names: List[str],
                             title: str = "Confusion Matrix",
                             save_path: Optional[str] = None) -> go.Figure:
        """Plot confusion matrix"""
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=600,
            height=600
        )
        
        if save_path:
            fig.write_html(str(self.output_dir / f"{save_path}.html"))
            fig.write_image(str(self.output_dir / f"{save_path}.png"))
        
        return fig
    
    def plot_training_history(self, history: Dict[str, List[float]],
                             title: str = "Training History",
                             save_path: Optional[str] = None) -> go.Figure:
        """Plot training history (loss, accuracy, etc.)"""
        
        # Create subplots
        metrics = list(history.keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            return go.Figure()
        
        # Determine subplot arrangement
        cols = min(2, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics):
            row = i // cols + 1
            col = i % cols + 1
            
            epochs = list(range(1, len(history[metric]) + 1))
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(color=self.colors[i % len(self.colors)])
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=title,
            height=300 * rows,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(str(self.output_dir / f"{save_path}.html"))
            fig.write_image(str(self.output_dir / f"{save_path}.png"))
        
        return fig
    
    def plot_dataset_overview(self, train_counts: Dict[str, int],
                             val_counts: Dict[str, int],
                             title: str = "Dataset Overview",
                             save_path: Optional[str] = None) -> go.Figure:
        """Plot dataset overview with train/validation split"""
        
        # Prepare data
        all_classes = list(set(train_counts.keys()) | set(val_counts.keys()))
        
        train_values = [train_counts.get(cls, 0) for cls in all_classes]
        val_values = [val_counts.get(cls, 0) for cls in all_classes]
        
        fig = go.Figure(data=[
            go.Bar(name='Training', x=all_classes, y=train_values),
            go.Bar(name='Validation', x=all_classes, y=val_values)
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Classes",
            yaxis_title="Number of Samples",
            barmode='group',
            height=500,
            xaxis_tickangle=-45
        )
        
        if save_path:
            fig.write_html(str(self.output_dir / f"{save_path}.html"))
            fig.write_image(str(self.output_dir / f"{save_path}.png"))
        
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]],
                             title: str = "Model Comparison",
                             save_path: Optional[str] = None) -> go.Figure:
        """Plot comparison between different models"""
        
        models = list(results.keys())
        metrics = list(results[models[0]].keys()) if models else []
        
        fig = go.Figure()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=values,
                yaxis=f'y{i+1}' if i > 0 else 'y',
                offsetgroup=i,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Models",
            barmode='group',
            height=500
        )
        
        if save_path:
            fig.write_html(str(self.output_dir / f"{save_path}.html"))
            fig.write_image(str(self.output_dir / f"{save_path}.png"))
        
        return fig
    
    def create_dashboard_report(self, 
                               class_distribution: Dict[str, int],
                               model_results: Dict[str, Any],
                               feature_importance: Optional[Tuple[List[str], List[float]]] = None,
                               save_path: str = "dashboard_report") -> str:
        """Create comprehensive dashboard report"""
        
        # Create multiple plots
        plots_html = []
        
        # 1. Class distribution
        fig1 = self.plot_class_distribution(class_distribution, "Dataset Class Distribution")
        plots_html.append(fig1.to_html(include_plotlyjs=False, div_id="class_dist"))
        
        # 2. Model comparison if available
        if 'model_comparison' in model_results:
            fig2 = self.plot_model_comparison(model_results['model_comparison'], "Model Performance Comparison")
            plots_html.append(fig2.to_html(include_plotlyjs=False, div_id="model_comp"))
        
        # 3. Feature importance if available
        if feature_importance:
            feature_names, importances = feature_importance
            fig3 = self.plot_feature_importance(feature_names, importances, "Top Feature Importance")
            plots_html.append(fig3.to_html(include_plotlyjs=False, div_id="feat_imp"))
        
        # Create HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Amulet-AI Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .plot-container {{ margin: 20px 0; }}
                h1, h2 {{ color: #2c3e50; }}
                .summary {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>🔮 Amulet-AI Analysis Dashboard</h1>
            <div class="summary">
                <h2>📊 Summary</h2>
                <p><strong>Total Classes:</strong> {len(class_distribution)}</p>
                <p><strong>Total Samples:</strong> {sum(class_distribution.values())}</p>
                <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            {''.join(f'<div class="plot-container">{plot}</div>' for plot in plots_html)}
            
            <div class="summary">
                <h2>🎯 Model Results</h2>
                <pre>{json.dumps(model_results, indent=2, ensure_ascii=False)}</pre>
            </div>
        </body>
        </html>
        """
        
        # Save dashboard
        dashboard_path = self.output_dir / f"{save_path}.html"
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(dashboard_path)

# Example usage and testing
if __name__ == "__main__":
    visualizer = CompatibleVisualizer()
    
    # Test data
    class_counts = {
        "พระสมเด็จ": 45,
        "พระสิวลี": 32,
        "พระพุทธเจ้า": 28,
        "สมเด็จแหวกม่าน": 25
    }
    
    # Test plots
    fig1 = visualizer.plot_class_distribution(class_counts, save_path="test_class_dist")
    print("✅ Class distribution plot created")
    
    # Test dashboard
    model_results = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88
    }
    
    dashboard_path = visualizer.create_dashboard_report(
        class_counts, 
        model_results,
        save_path="test_dashboard"
    )
    print(f"✅ Dashboard created: {dashboard_path}")