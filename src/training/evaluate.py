"""
Evaluate trained model on test set.
Compute detailed metrics and visualizations.
"""

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from tqdm import tqdm


class ModelEvaluator:
    """Evaluate YOLO model performance."""
    
    def __init__(self, model_path, data_yaml, output_dir='results/evaluation'):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model weights
            data_yaml: Path to dataset YAML
            output_dir: Directory to save evaluation results
        """
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"📥 Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        
        self.class_names = ['pothole', 'longitudinal_crack', 'crazing', 'faded_marking']
    
    def evaluate(self, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run comprehensive evaluation.
        
        Args:
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        print("🔍 Running evaluation...")
        
        # Run validation
        metrics = self.model.val(
            data=str(self.data_yaml),
            conf=conf_threshold,
            iou=iou_threshold,
            plots=True,
            save_json=True
        )
        
        # Compile metrics
        results = {
            'overall_metrics': {
                'mAP50': float(metrics.box.map50),
                'mAP50_95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
                'f1_score': float(2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6))
            },
            'per_class_metrics': {}
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            if i < len(metrics.box.maps):
                results['per_class_metrics'][class_name] = {
                    'AP50': float(metrics.box.maps[i]),
                    'precision': float(metrics.box.p[i]) if hasattr(metrics.box, 'p') else 0.0,
                    'recall': float(metrics.box.r[i]) if hasattr(metrics.box, 'r') else 0.0,
                }
        
        # Save results
        results_file = self.output_dir / 'evaluation_metrics.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {results_file}")
        
        # Print summary
        self._print_summary(results)
        
        # Generate visualizations
        self._plot_metrics(results)
        
        return results
    
    def _print_summary(self, results):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        print("\n🎯 Overall Metrics:")
        for metric, value in results['overall_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n📈 Per-Class Metrics:")
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"\n  {class_name}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
        
        print("\n" + "="*60)
    
    def _plot_metrics(self, results):
        """Generate metric visualizations."""
        print("📊 Generating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Per-class AP50 bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = list(results['per_class_metrics'].keys())
        ap_values = [results['per_class_metrics'][c]['AP50'] for c in classes]
        
        bars = ax.bar(classes, ap_values, color='steelblue', alpha=0.8)
        ax.set_ylabel('AP@0.5', fontsize=12)
        ax.set_title('Average Precision per Class', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_ap_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        precision_values = [results['per_class_metrics'][c]['precision'] for c in classes]
        recall_values = [results['per_class_metrics'][c]['recall'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, precision_values, width, label='Precision', color='coral', alpha=0.8)
        bars2 = ax.bar(x + width/2, recall_values, width, label='Recall', color='lightgreen', alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision vs Recall per Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim([0, 1])
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Overall metrics radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        metrics_names = list(results['overall_metrics'].keys())
        metrics_values = list(results['overall_metrics'].values())
        
        # Number of variables
        N = len(metrics_names)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        metrics_values += metrics_values[:1]
        angles += angles[:1]
        
        ax.plot(angles, metrics_values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, metrics_values, alpha=0.25, color='steelblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names, size=10)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Model Performance', size=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_metrics_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {self.output_dir}")
    
    def analyze_predictions(self, test_images_dir, num_samples=10):
        """
        Analyze predictions on sample images.
        
        Args:
            test_images_dir: Directory with test images
            num_samples: Number of samples to analyze
        """
        print(f"🔍 Analyzing predictions on {num_samples} samples...")
        
        test_images_dir = Path(test_images_dir)
        image_files = list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png'))
        
        if len(image_files) == 0:
            print("⚠️ No images found in test directory")
            return
        
        # Sample random images
        sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        # Create output directory
        predictions_dir = self.output_dir / 'sample_predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        # Run predictions
        for img_path in tqdm(sample_files):
            # Predict
            results = self.model.predict(
                source=str(img_path),
                conf=0.25,
                iou=0.45,
                save=False
            )
            
            # Visualize
            result = results[0]
            img_with_boxes = result.plot()
            
            # Save
            output_path = predictions_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(output_path), img_with_boxes)
        
        print(f"✅ Sample predictions saved to {predictions_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML')
    parser.add_argument('--output', type=str, default='results/evaluation', 
                        help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--test-images', type=str, default=None, 
                        help='Path to test images for sample predictions')
    parser.add_argument('--num-samples', type=int, default=10, 
                        help='Number of sample predictions')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        model_path=args.model,
        data_yaml=args.data,
        output_dir=args.output
    )
    
    # Run evaluation
    evaluator.evaluate(
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Analyze sample predictions
    if args.test_images:
        evaluator.analyze_predictions(
            test_images_dir=args.test_images,
            num_samples=args.num_samples
        )


if __name__ == "__main__":
    main()
