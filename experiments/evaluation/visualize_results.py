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

# Add project root to path (go up 3 levels from file: evaluation -> experiments -> project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from config import *

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

os.makedirs(PLOTS_DIR, exist_ok=True)
HEADLESS = bool(int(os.environ.get("HEADLESS", "0")))

def load_evaluation_results():
    results_file = os.path.join(METRICS_DIR, 'evaluation_results.json')
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return None
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read results file: {e}")
        return None

def plot_comprehensive_comparison(results):
    try:
        secure_errors = np.array(results['secure_model']['errors'], dtype=float)
        attacked_errors = np.array(results['attacked_model']['errors'], dtype=float)
        secure_metrics = results['secure_model'].get('metrics', {})
        attacked_metrics = results['attacked_model'].get('metrics', {})
    except Exception as e:
        print("Invalid results structure for plotting"); return

    n = len(secure_errors)
    if n == 0 or len(attacked_errors) == 0:
        print("Empty error arrays — skipping comprehensive plot"); return

    fig = plt.figure(figsize=(18, 12))
    frames = np.arange(n)
    secure_color = '#2ecc71'
    attacked_color = '#e74c3c'

    # Panel 1
    try:
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(frames, secure_errors, label='FedGuard (Secure)', color=secure_color, linewidth=2, alpha=0.85)
        ax1.plot(frames, attacked_errors, label='Attacked (No Defense)', color=attacked_color, linewidth=2, linestyle='--', alpha=0.85)
        secure_threshold = secure_metrics.get('threshold', None)
        if secure_threshold is not None:
            ax1.axhline(y=secure_threshold, color='gray', linestyle=':', label=f'Threshold ({secure_threshold:.4f})', alpha=0.6)
        ax1.set_xlabel('Frame Number'); ax1.set_ylabel('Reconstruction Error (MSE)')
        ax1.set_title('A) Reconstruction Error Comparison'); ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.3)
    except Exception:
        print("Panel 1 failed")

    # Panel 2
    try:
        ax2 = plt.subplot(3, 3, 2)
        ax2.hist(secure_errors, bins=50, alpha=0.6, color=secure_color, label='Secure', edgecolor='black', linewidth=0.5)
        ax2.hist(attacked_errors, bins=50, alpha=0.6, color=attacked_color, label='Attacked', edgecolor='black', linewidth=0.5)
        ax2.set_title('B) Error Distribution'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')
    except Exception:
        print("Panel 2 failed")

    # Panel 3
    try:
        ax3 = plt.subplot(3, 3, 3)
        error_diff = attacked_errors - secure_errors
        colors = [attacked_color if d > 0 else secure_color for d in error_diff]
        ax3.bar(frames, error_diff, color=colors, alpha=0.6, width=1)
        ax3.axhline(y=0, color='black', linewidth=1)
        ax3.set_title('C) Impact of Attack'); ax3.grid(True, alpha=0.3, axis='y')
    except Exception:
        print("Panel 3 failed")

    # Panel 4
    try:
        ax4 = plt.subplot(3, 3, 4)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        secure_vals = [secure_metrics.get('accuracy',0), secure_metrics.get('precision',0), secure_metrics.get('recall',0), secure_metrics.get('f1_score',0), secure_metrics.get('auc',0)]
        attacked_vals = [attacked_metrics.get('accuracy',0), attacked_metrics.get('precision',0), attacked_metrics.get('recall',0), attacked_metrics.get('f1_score',0), attacked_metrics.get('auc',0)]
        x = np.arange(len(metrics_names)); width = 0.35
        bars1 = ax4.bar(x - width/2, secure_vals, width, label='Secure', color=secure_color, edgecolor='black')
        bars2 = ax4.bar(x + width/2, attacked_vals, width, label='Attacked', color=attacked_color, edgecolor='black')
        ax4.set_xticks(x); ax4.set_xticklabels(metrics_names, rotation=45, ha='right'); ax4.set_ylim([0,1.05]); ax4.legend()
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    except Exception:
        print("Panel 4 failed")

    # Panel 5 & 6
    try:
        ax5 = plt.subplot(3, 3, 5)
        cm_secure = np.array([[secure_metrics.get('true_negatives',0), secure_metrics.get('false_positives',0)],[secure_metrics.get('false_negatives',0), secure_metrics.get('true_positives',0)]])
        sns.heatmap(cm_secure, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'], ax=ax5, cbar_kws={'label':'Count'})
        ax5.set_title('E) Confusion Matrix (Secure)')
    except Exception:
        print("Panel 5 failed")
    try:
        ax6 = plt.subplot(3, 3, 6)
        cm_attacked = np.array([[attacked_metrics.get('true_negatives',0), attacked_metrics.get('false_positives',0)],[attacked_metrics.get('false_negatives',0), attacked_metrics.get('true_positives',0)]])
        sns.heatmap(cm_attacked, annot=True, fmt='d', cmap='Reds', xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'], ax=ax6, cbar_kws={'label':'Count'})
        ax6.set_title('F) Confusion Matrix (Attacked)')
    except Exception:
        print("Panel 6 failed")

    # Panel 7
    try:
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot([0,1],[0,1],'k--', linewidth=1)
        sec_prec = secure_metrics.get('precision',0); sec_rec = secure_metrics.get('recall',0)
        att_prec = attacked_metrics.get('precision',0); att_rec = attacked_metrics.get('recall',0)
        ax7.scatter([1-sec_prec],[sec_rec], s=200, color=secure_color, marker='o', label=f"Secure (AUC={secure_metrics.get('auc',0.0):.3f})", edgecolors='black')
        ax7.scatter([1-att_prec],[att_rec], s=200, color=attacked_color, marker='s', label=f"Attacked (AUC={attacked_metrics.get('auc',0.0):.3f})", edgecolors='black')
        ax7.set_xlim([-0.05,1.05]); ax7.set_ylim([-0.05,1.05]); ax7.legend()
    except Exception:
        print("Panel 7 failed")

    # Panel 8
    try:
        ax8 = plt.subplot(3, 3, 8)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        secure_vals = [secure_metrics.get('accuracy',0), secure_metrics.get('precision',0), secure_metrics.get('recall',0), secure_metrics.get('f1_score',0), secure_metrics.get('auc',0)]
        attacked_vals = [attacked_metrics.get('accuracy',0), attacked_metrics.get('precision',0), attacked_metrics.get('recall',0), attacked_metrics.get('f1_score',0), attacked_metrics.get('auc',0)]
        improvements = [((s - a) / (a + 1e-8)) * 100 for s,a in zip(secure_vals, attacked_vals)]
        colors_imp = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        bars = ax8.barh(metrics_names, improvements, color=colors_imp, edgecolor='black')
        ax8.axvline(x=0, color='black', linewidth=1)
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            ax8.text(val, i, f' {val:+.1f}%', va='center', ha='left' if val>0 else 'right')
    except Exception:
        print("Panel 8 failed")

    # Panel 9
    try:
        ax9 = plt.subplot(3, 3, 9); ax9.axis('off')
        avg_imp = float(np.mean(improvements)) if len(improvements)>0 else 0.0
        best_imp = float(np.max(improvements)) if len(improvements)>0 else 0.0
        summary_text = f"FEDGUARD EVALUATION SUMMARY\n\nSecure Model (FedGuard):\n • Accuracy: {secure_metrics.get('accuracy',0.0):.2%}\n • Precision: {secure_metrics.get('precision',0.0):.2%}\n • Recall: {secure_metrics.get('recall',0.0):.2%}\n • F1-Score: {secure_metrics.get('f1_score',0.0):.4f}\n • AUC-ROC: {secure_metrics.get('auc',0.0):.4f}\n\nAttacked Model (No Defense):\n • Accuracy: {attacked_metrics.get('accuracy',0.0):.2%}\n • Precision: {attacked_metrics.get('precision',0.0):.2%}\n • Recall: {attacked_metrics.get('recall',0.0):.2%}\n • F1-Score: {attacked_metrics.get('f1_score',0.0):.4f}\n • AUC-ROC: {attacked_metrics.get('auc',0.0):.4f}\n\nOverall Improvement:\n • Avg: {avg_imp:+.1f}%\n • Best: {best_imp:+.1f}%\n\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ax9.text(0.02, 0.98, summary_text, transform=ax9.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    except Exception:
        print("Panel 9 failed")

    fig.suptitle('FedGuard: Secure Federated Learning Evaluation Results', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0,0,1,0.98])
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(PLOTS_DIR, f'comprehensive_evaluation_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comprehensive plot saved: {save_path}")
    except Exception:
        print("Failed to save comprehensive plot")
    if not HEADLESS:
        plt.show()
    else:
        plt.close(fig)

def plot_simple_comparison(results):
    secure_errors = np.array(results['secure_model']['errors'], dtype=float)
    attacked_errors = np.array(results['attacked_model']['errors'], dtype=float)
    if secure_errors.size == 0 or attacked_errors.size == 0:
        print("Empty error arrays — skipping simple comparison plot."); return

    plt.figure(figsize=(14, 7))
    frames = np.arange(len(secure_errors))
    plt.plot(frames, secure_errors, label='FedGuard (Secure Model)', color='#2ecc71', linewidth=2.5)
    plt.plot(frames, attacked_errors, label='Attacked Model (No Defense)', color='#e74c3c', linewidth=2.5, linestyle='--')
    threshold = results['secure_model'].get('metrics', {}).get('threshold', None)
    if threshold is not None:
        plt.axhline(y=threshold, color='gray', linestyle=':', linewidth=2, label=f'Anomaly Threshold ({threshold:.4f})', alpha=0.7)
    plt.title('FedGuard vs. Attacked Model: Anomaly Detection Performance')
    plt.xlabel('Video Frame Number'); plt.ylabel('Anomaly Score (Reconstruction Error)')
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    max_err = float(np.nanmax(np.concatenate([secure_errors, attacked_errors])))
    if threshold is not None:
        plt.fill_between(frames, threshold, max_err, alpha=0.08, color='red')
    plt.tight_layout()
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(PLOTS_DIR, f'simple_comparison_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Simple comparison plot saved: {save_path}")
    except Exception:
        print("Failed to save simple comparison plot")
    if not HEADLESS:
        plt.show()
    else:
        plt.close()

def main():
    print("\nFEDGUARD VISUALIZATION\n")
    results = load_evaluation_results()
    if results is None:
        return
    print("Generating visualizations...")
    plot_comprehensive_comparison(results)
    plot_simple_comparison(results)
    print("\n✅ ALL VISUALIZATIONS COMPLETE")
    print(f"\nPlots saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
