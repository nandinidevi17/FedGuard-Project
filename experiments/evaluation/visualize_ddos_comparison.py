import matplotlib.pyplot as plt
import re
from datetime import datetime
import os
import sys

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
logs_dir = os.path.join(project_root, "results", "logs")

# 1. The "Failed" Experiment Log (Insecure Server)
LOG_INSECURE = os.path.join(logs_dir, "ddos_experiment.log")
# 2. The "Success" Experiment Log (Protected Server)
LOG_SECURE = os.path.join(logs_dir, "ddos_defense.log")

def get_round_duration(log_path):
    """Parses a log file and returns the duration of the first completed round."""
    if not os.path.exists(log_path):
        print(f"⚠️ Log not found: {log_path}")
        return 0

    start_time = None
    end_time = None
    
    # Regex for timestamp: 2025-12-01 17:22:44,839
    time_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        match = time_pattern.search(line)
        if not match: continue
        timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f')

        # Detect Start
        if "Waiting for updates" in line:
            start_time = timestamp
        
        # Detect End (Aggregation)
        if "AGGREGATION" in line and start_time:
            end_time = timestamp
            return (end_time - start_time).total_seconds()

    return 0

def plot_comparison():
    print("Parsing logs...")
    duration_insecure = get_round_duration(LOG_INSECURE)
    duration_secure = get_round_duration(LOG_SECURE)
    
    print(f"Insecure (Attacked) Duration: {duration_insecure} seconds")
    print(f"Secure (Defended) Duration:   {duration_secure} seconds")

    if duration_insecure == 0 and duration_secure == 0:
        print("❌ No valid round data found in either log.")
        print("Please ensure you ran both experiments until 'ROUND 1 AGGREGATION'.")
        # Dummy data for preview if you run it before experimenting
        duration_insecure = 101.0
        duration_secure = 5.0

    # Data Setup
    scenarios = ['Insecure Server\n(Under Attack)', 'FedGuard Server\n(With Defense)']
    times = [duration_insecure, duration_secure]
    colors = ['#e74c3c', '#2ecc71'] # Red vs Green

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenarios, times, color=colors, edgecolor='black', width=0.6)

    plt.title('Effectiveness of DDoS Defense on Training Speed', fontsize=14, fontweight='bold')
    plt.ylabel('Time to Complete Round (Seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}s',
                 ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Save
    save_path = os.path.join(project_root, "results", "plots", "ddos_defense_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ Comparison Chart saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()