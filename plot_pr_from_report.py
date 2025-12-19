"""
CREATE PRECISION-RECALL CURVES FROM EXISTING EVALUATION DATA
=============================================================
This script generates PR curves using the classification report data
from your existing evaluation results, avoiding TensorFlow loading issues.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Define all sign language letters (A-Z)
actions = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


def load_classification_report(report_path):
    """Load classification report JSON"""
    with open(report_path, 'r') as f:
        return json.load(f)


def estimate_pr_curves_from_metrics(report, save_dir="pr_curves_from_report"):
    """
    Estimate precision-recall relationships from classification metrics.
    While we can't generate exact PR curves without predictions, we can
    visualize the precision and recall for each class at their decision threshold.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("ANALYZING PRECISION-RECALL METRICS")
    print("="*70)
    
    # Extract per-class metrics
    class_metrics = {}
    for action in actions:
        if action in report:
            class_metrics[action] = {
                'precision': report[action]['precision'],
                'recall': report[action]['recall'],
                'f1-score': report[action]['f1-score'],
                'support': report[action]['support']
            }
            print(f"Class {action}: P={class_metrics[action]['precision']:.4f}, "
                  f"R={class_metrics[action]['recall']:.4f}, "
                  f"F1={class_metrics[action]['f1-score']:.4f}")
    
    # === PLOT 1: PRECISION VS RECALL SCATTER ===
    plt.figure(figsize=(14, 10))
    
    precisions = [class_metrics[a]['precision'] for a in actions if a in class_metrics]
    recalls = [class_metrics[a]['recall'] for a in actions if a in class_metrics]
    f1_scores = [class_metrics[a]['f1-score'] for a in actions if a in class_metrics]
    
    # Create scatter plot with color based on F1-score
    scatter = plt.scatter(recalls, precisions, c=f1_scores, cmap='viridis',
                         s=300, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels for each point
    for i, action in enumerate([a for a in actions if a in class_metrics]):
        plt.annotate(action, (recalls[i], precisions[i]),
                    fontsize=12, fontweight='bold',
                    ha='center', va='center')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('F1-Score', fontsize=12, fontweight='bold')
    
    # Add diagonal line for F1 reference
    x = np.linspace(0, 1, 100)
    for f1 in [0.5, 0.7, 0.9]:
        y = (f1 * x) / (2 * x - f1)
        y = np.clip(y, 0, 1)
        plt.plot(x, y, '--', alpha=0.3, label=f'F1={f1:.1f}')
    
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Precision vs Recall for Each Class', fontsize=16, fontweight='bold', pad=20)
    plt.xlim([0.85, 1.05])
    plt.ylim([0.85, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "precision_recall_scatter.png"), dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {save_dir}/precision_recall_scatter.png")
    plt.close()
    
    # === PLOT 2: PRECISION-RECALL BAR CHART ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # First half (A-M)
    class_names_am = actions[:13]
    precisions_am = [class_metrics[a]['precision'] for a in class_names_am if a in class_metrics]
    recalls_am = [class_metrics[a]['recall'] for a in class_names_am if a in class_metrics]
    
    x = np.arange(len(class_names_am))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, precisions_am, width, label='Precision',
                    color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, recalls_am, width, label='Recall',
                    color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Sign Letter', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Precision & Recall (A-M)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names_am)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0.85, 1.05])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Second half (N-Z)
    class_names_nz = actions[13:]
    precisions_nz = [class_metrics[a]['precision'] for a in class_names_nz if a in class_metrics]
    recalls_nz = [class_metrics[a]['recall'] for a in class_names_nz if a in class_metrics]
    
    x = np.arange(len(class_names_nz))
    
    bars1 = ax2.bar(x - width/2, precisions_nz, width, label='Precision',
                    color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, recalls_nz, width, label='Recall',
                    color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Sign Letter', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Precision & Recall (N-Z)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names_nz)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0.85, 1.05])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "precision_recall_bars.png"), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/precision_recall_bars.png")
    plt.close()
    
    # === PLOT 3: F1-SCORE COMPARISON ===
    plt.figure(figsize=(16, 8))
    
    f1_scores_all = [class_metrics[a]['f1-score'] for a in actions if a in class_metrics]
    colors_bar = ['#2ecc71' if f1 >= 0.95 else '#f39c12' if f1 >= 0.9 else '#e74c3c' 
                  for f1 in f1_scores_all]
    
    bars = plt.bar(actions, f1_scores_all, color=colors_bar, edgecolor='black', linewidth=1.5)
    
    # Add average line
    avg_f1 = np.mean(f1_scores_all)
    plt.axhline(y=avg_f1, color='navy', linestyle='--', linewidth=2,
                label=f'Average F1 ({avg_f1:.3f})')
    
    # Add value labels
    for bar, f1 in zip(bars, f1_scores_all):
        plt.text(bar.get_x() + bar.get_width()/2., f1 + 0.005,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Sign Letter', fontsize=13, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=13, fontweight='bold')
    plt.title('F1-Score per Class (Harmonic Mean of Precision & Recall)',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylim([0.85, 1.05])
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "f1_scores_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/f1_scores_comparison.png")
    plt.close()
    
    # === PLOT 4: SUMMARY TABLE VISUALIZATION ===
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = [['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]
    for action in actions:
        if action in class_metrics:
            table_data.append([
                action,
                f"{class_metrics[action]['precision']:.4f}",
                f"{class_metrics[action]['recall']:.4f}",
                f"{class_metrics[action]['f1-score']:.4f}",
                str(class_metrics[action]['support'])
            ])
    
    # Add summary rows
    if 'macro avg' in report:
        table_data.append(['---', '---', '---', '---', '---'])
        table_data.append([
            'Macro Avg',
            f"{report['macro avg']['precision']:.4f}",
            f"{report['macro avg']['recall']:.4f}",
            f"{report['macro avg']['f1-score']:.4f}",
            str(report['macro avg']['support'])
        ])
    
    if 'weighted avg' in report:
        table_data.append([
            'Weighted Avg',
            f"{report['weighted avg']['precision']:.4f}",
            f"{report['weighted avg']['recall']:.4f}",
            f"{report['weighted avg']['f1-score']:.4f}",
            str(report['weighted avg']['support'])
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.2, 0.2, 0.2, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows with alternating colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if table_data[i][0] == '---':
                table[(i, j)].set_facecolor('#ecf0f1')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Detailed Classification Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, "metrics_table.png"), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/metrics_table.png")
    plt.close()
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    
    # Print summary statistics
    print(f"\nAverage Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f}")
    print(f"Average F1-Score: {avg_f1:.4f}")
    
    # Find best and worst
    f1_dict = {a: class_metrics[a]['f1-score'] for a in actions if a in class_metrics}
    sorted_f1 = sorted(f1_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🏆 Top 5 Best F1-Scores:")
    for i, (cls, f1) in enumerate(sorted_f1[:5], 1):
        print(f"  {i}. {cls}: {f1:.4f}")
    
    print("\n⚠️  Top 5 Worst F1-Scores:")
    for i, (cls, f1) in enumerate(sorted_f1[-5:][::-1], 1):
        print(f"  {i}. {cls}: {f1:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PR visualizations from classification report")
    parser.add_argument("--report", default="evaluation_results/classification_report.json",
                       help="Path to classification report JSON")
    parser.add_argument("--output_dir", default="pr_curves_from_report",
                       help="Output directory for plots")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PRECISION-RECALL VISUALIZATION FROM CLASSIFICATION REPORT")
    print("="*70)
    
    if not os.path.exists(args.report):
        print(f"\n❌ ERROR: Classification report not found at {args.report}")
        print("\nPlease run evaluate_model.py first to generate the report.")
        return
    
    print(f"\nLoading classification report from: {args.report}")
    report = load_classification_report(args.report)
    
    estimate_pr_curves_from_metrics(report, args.output_dir)
    
    print(f"\n💾 All visualizations saved to: {args.output_dir}/\n")


if __name__ == "__main__":
    main()
