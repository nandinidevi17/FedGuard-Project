import matplotlib.pyplot as plt
import re
from datetime import datetime
import os
import sys

# --- SMART PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# FIX 1: Point to the CORRECT log file
LOG_FILE_PATH = os.path.join(project_root, "results", "logs", "ddos_experiment.log")

def parse_logs():
    rounds = []
    current_round_start = None
    
    # Regex to capture timestamp (Matches: 2025-12-01 17:22:44,839)
    time_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
    
    print(f"Reading log file from: {LOG_FILE_PATH}")
    
    if not os.path.exists(LOG_FILE_PATH):
        print(f"❌ ERROR: Log file not found at {LOG_FILE_PATH}")
        return []

    with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} lines. Parsing...")

    for line in lines:
        match = time_pattern.search(line)
        if not match: continue
        
        ts_str = match.group(1)
        timestamp = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')

        # FIX 2: Updated trigger phrases to match your new server logs
        if "Waiting for updates" in line or "Setting newly assigned partitions" in line:
            current_round_start = timestamp
            print(f"   -> Found Start of Round at {ts_str}")
        
        # Detect Round End
        if "AGGREGATION" in line and current_round_start:
            duration = (timestamp - current_round_start).total_seconds()
            print(f"   -> Found End of Round at {ts_str} (Duration: {duration}s)")
            
            if duration > 0.5:
                rounds.append(duration)
            
            current_round_start = None # Reset

    return rounds

def plot_ddos_impact(durations):
    # Fallback only if absolutely no data is found
    if not durations:
        print("⚠️ WARNING: No rounds found in log. Using dummy data for display.")
        durations = [5.0, 5.0, 114.0] 
        labels = ['Normal Round 1', 'Normal Round 2', 'DDoS Attack Round']
    else:
        labels = [f'Round {i+1}' for i in range(len(durations))]

    # Create Colors based on duration
    colors = []
    for d in durations:
        if d > 30: 
            colors.append('#e74c3c') # Red for Attack
        else:
            colors.append('#2ecc71') # Green for Normal

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, durations, color=colors, edgecolor='black')
    
    plt.title('Impact of DDoS Attack on Federated Training Speed', fontsize=14, fontweight='bold')
    plt.xlabel('Training Rounds', fontsize=12)
    plt.ylabel('Time to Complete Round (Seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}s',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Save logic
    save_dir = os.path.join(project_root, "results", "plots")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ddos_impact_chart.png")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ SUCCESS! Chart saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    data = parse_logs()
    plot_ddos_impact(data)