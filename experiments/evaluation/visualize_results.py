"""
Advanced visualization of evaluation results.
Creates publication-quality plots comparing secure vs attacked models.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


def load_evaluation_results():
    """Load results from evaluation."""
    results_file = os.path.join(METRICS_DIR, 'evaluation_results.json')
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("Please run evaluate_model.py first!")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_comprehensive_comparison(results):
    """
    Create a comprehensive multi-panel figure.
    """
    fig = plt.figure(figsize=(18, 12))
    
    secure_errors = results['secure_model']['errors']
    attacked_errors = results['attacked_model']['errors']
    secure_metrics = results['secure_model']['metrics']
    attacked_metrics = results['attacked_model']['metrics']
    
    # Panel 1: Reconstruction Errors Over Time
    ax1 = plt.subplot(3, 3, 1)
    frames = range(len(secure_errors))
    ax1.plot(frames, secure_errors, label='FedGuard (Secure)', 
             color='#2ecc71', linewidth=2, alpha=0.8)
    ax1.plot(frames, attacked_errors, label='Attacked (No Defense)', 
             color='#e74c3c', linewidth=2, alpha=0.8, linestyle='--')
    
    # Highlight anomaly regions (if ground truth available)
    secure_threshold = secure_metrics['threshold']
    ax1.axhline(y=secure_threshold, color='gray', linestyle=':', 
                label=f'Threshold ({secure_threshold:.4f})', alpha=0.5)
    
    ax1.set_xlabel('Frame Number', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Reconstruction Error (MSE)', fontsize=11, fontweight='bold')
    ax1.set_title('A) Reconstruction Error Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Error Distribution (Histogram)
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(secure_errors, bins=50, alpha=0.6, color='#2ecc71', 
             label='Secure', edgecolor='black', linewidth=0.5)
    ax2.hist(attacked_errors, bins=50, alpha=0.6, color='#e74c3c', 
             label='Attacked', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Reconstruction Error', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('B) Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Error Difference
    ax3 = plt.subplot(3, 3, 3)
    error_diff = np.array(attacked_errors) - np.array(secure_errors)
    colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in error_diff]
    ax3.bar(frames, error_diff, color=colors, alpha=0.6, width=1)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_xlabel('Frame Number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Error Difference\n(Attacked - Secure)', fontsize=11, fontweight='bold')
    ax3.set_title('C) Impact of Attack', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Metrics Comparison (Bar Chart)
    ax4 = plt.subplot(3, 3, 4)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    secure_vals = [
        secure_metrics['accuracy'],
        secure_metrics['precision'],
        secure_metrics['recall'],
        secure_metrics['f1_score'],
        secure_metrics['auc']
    ]
    attacked_vals = [
        attacked_metrics['accuracy'],
        attacked_metrics['precision'],
        attacked_metrics['recall'],
        attacked_metrics['f1_score'],
        attacked_metrics['auc']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, secure_vals, width, label='Secure', 
                    color='#2ecc71', edgecolor='black', linewidth=1)
    bars2 = ax4.bar(x + width/2, attacked_vals, width, label='Attacked', 
                    color='#e74c3c', edgecolor='black', linewidth=1)
    
    ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax4.set_title('D) Performance Metrics', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names, rotation=45, ha='right', fontsize=9)
    ax4.legend(fontsize=9)
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Panel 5: Confusion Matrix (Secure)
    ax5 = plt.subplot(3, 3, 5)
    cm_secure = np.array([
        [secure_metrics['true_negatives'], secure_metrics['false_positives']],
        [secure_metrics['false_negatives'], secure_metrics['true_positives']]
    ])
    sns.heatmap(cm_secure, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'], ax=ax5,
                cbar_kws={'label': 'Count'})
    ax5.set_title('E) Confusion Matrix (Secure)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    # Panel 6: Confusion Matrix (Attacked)
    ax6 = plt.subplot(3, 3, 6)
    cm_attacked = np.array([
        [attacked_metrics['true_negatives'], attacked_metrics['false_positives']],
        [attacked_metrics['false_negatives'], attacked_metrics['true_positives']]
    ])
    sns.heatmap(cm_attacked, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'], ax=ax6,
                cbar_kws={'label': 'Count'})
    ax6.set_title('F) Confusion Matrix (Attacked)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    # Panel 7: ROC Curves (if available in results)
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax7.plot([0, 0, 1], [0, 1, 1], 'gray', linewidth=1, alpha=0.5, 
             label='Perfect Classifier')
    
    # Note: ROC curves would need FPR/TPR from results
    ax7.scatter([1 - secure_metrics['precision']], [secure_metrics['recall']], 
               s=200, color='#2ecc71', marker='o', 
               label=f"Secure (AUC={secure_metrics['auc']:.3f})", 
               edgecolors='black', linewidth=2, zorder=5)
    ax7.scatter([1 - attacked_metrics['precision']], [attacked_metrics['recall']], 
               s=200, color='#e74c3c', marker='s', 
               label=f"Attacked (AUC={attacked_metrics['auc']:.3f})",
               edgecolors='black', linewidth=2, zorder=5)
    
    ax7.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax7.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax7.set_title('G) ROC Space', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9, loc='lower right')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([-0.05, 1.05])
    ax7.set_ylim([-0.05, 1.05])
    
    # Panel 8: Improvement Percentages
    ax8 = plt.subplot(3, 3, 8)
    improvements = []
    for i, metric in enumerate(metrics_names):
        secure_val = secure_vals[i]
        attacked_val = attacked_vals[i]
        improvement = ((secure_val - attacked_val) / (attacked_val + 1e-8)) * 100
        improvements.append(improvement)
    
    colors_imp = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax8.barh(metrics_names, improvements, color=colors_imp, 
                    edgecolor='black', linewidth=1)
    ax8.axvline(x=0, color='black', linewidth=1)
    ax8.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax8.set_title('H) FedGuard Improvement', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax8.text(val, i, f' {val:+.1f}%', va='center', 
                ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')
    
    # Panel 9: Statistics Summary (Text)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    FEDGUARD EVALUATION SUMMARY
    
    Secure Model (FedGuard):
    • Accuracy: {secure_metrics['accuracy']:.2%}
    • Precision: {secure_metrics['precision']:.2%}
    • Recall: {secure_metrics['recall']:.2%}
    • F1-Score: {secure_metrics['f1_score']:.4f}
    • AUC-ROC: {secure_metrics['auc']:.4f}
    
    Attacked Model (No Defense):
    • Accuracy: {attacked_metrics['accuracy']:.2%}
    • Precision: {attacked_metrics['precision']:.2%}
    • Recall: {attacked_metrics['recall']:.2%}
    • F1-Score: {attacked_metrics['f1_score']:.4f}
    • AUC-ROC: {attacked_metrics['auc']:.4f}
    
    Overall Improvement:
    • Avg: {np.mean(improvements):+.1f}%
    • Best: {max(improvements):+.1f}%
    
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """
    
    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    fig.suptitle('FedGuard: Secure Federated Learning Evaluation Results', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(PLOTS_DIR, f'comprehensive_evaluation_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive plot saved: {save_path}")
    
    plt.show()


def plot_simple_comparison(results):
    """
    Create the original simple comparison plot (improved version).
    """
    secure_errors = results['secure_model']['errors']
    attacked_errors = results['attacked_model']['errors']
    
    plt.figure(figsize=(14, 7))
    
    frames = range(len(secure_errors))
    
    # Plot both lines
    plt.plot(frames, secure_errors, label='FedGuard (Secure Model)', 
             color='#2ecc71', linewidth=2.5, alpha=0.9)
    plt.plot(frames, attacked_errors, label='Attacked Model (No Defense)', 
             color='#e74c3c', linewidth=2.5, linestyle='--', alpha=0.9)
    
    # Add threshold line
    threshold = results['secure_model']['metrics']['threshold']
    plt.axhline(y=threshold, color='gray', linestyle=':', linewidth=2,
                label=f'Anomaly Threshold ({threshold:.4f})', alpha=0.7)
    
    # Styling
    plt.title('FedGuard vs. Attacked Model: Anomaly Detection Performance', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Video Frame Number', fontsize=13, fontweight='bold')
    plt.ylabel('Anomaly Score (Reconstruction Error)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add shaded region above threshold
    plt.fill_between(frames, threshold, max(max(secure_errors), max(attacked_errors)),
                     alpha=0.1, color='red', label='_nolegend_')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(PLOTS_DIR, f'simple_comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Simple comparison plot saved: {save_path}")
    
    plt.show()


def main():
    """Main visualization function."""
    print("\n" + "="*60)
    print("FEDGUARD VISUALIZATION")
    print("="*60)
    
    # Load results
    print("\nLoading evaluation results...")
    results = load_evaluation_results()
    
    if results is None:
        return
    
    print("✓ Results loaded successfully")
    
    # Generate plots
    print("\nGenerating visualizations...")
    print("  1. Creating comprehensive multi-panel figure...")
    plot_comprehensive_comparison(results)
    
    print("  2. Creating simple comparison plot...")
    plot_simple_comparison(results)
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETE")
    print("="*60)
    print(f"\nPlots saved in: {PLOTS_DIR}")
    print("\nYou now have:")
    print("  ✓ Comprehensive 9-panel analysis")
    print("  ✓ Simple comparison plot")
    print("  ✓ High-resolution images (300 DPI)")


if __name__ == "__main__":
    main()