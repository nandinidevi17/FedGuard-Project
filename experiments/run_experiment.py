"""
Automated experiment runner.
Runs complete FedGuard experiment end-to-end.
"""
import os
import sys
import subprocess
import time
import logging
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'experiment_runner.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Set UTF-8 encoding for stdout to handle emojis
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Automated experiment orchestration."""
    
    def __init__(self, experiment_name="fedguard_experiment"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(RESULTS_DIR, f"{experiment_name}_{self.timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        logger.info("="*60)
        logger.info(f"EXPERIMENT: {experiment_name}")
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info(f"Output directory: {self.experiment_dir}")
        logger.info("="*60)
    
    def check_prerequisites(self):
        """Check if everything is ready."""
        logger.info("\n🔍 Checking prerequisites...")
        
        # Check if baseline model exists
        baseline_path = os.path.join(MODELS_DIR, 'ucsd_baseline.h5')
        if not os.path.exists(baseline_path):
            logger.error(f"❌ Baseline model not found: {baseline_path}")
            logger.error("Please run training/train_baseline_model.py first!")
            return False
        
        logger.info("✓ Baseline model found")
        
        # Check if Kafka is running (try to connect)
        try:
            from kafka import KafkaProducer
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_SERVER,
                request_timeout_ms=5000
            )
            producer.close()
            logger.info("✓ Kafka server is running")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Kafka: {str(e)}")
            logger.error("Please start Kafka server first!")
            return False
        
        # Check if dataset exists
        if not os.path.exists(UCSD_PED1_TRAIN):
            logger.error(f"❌ Dataset not found: {UCSD_PED1_TRAIN}")
            return False
        
        logger.info("✓ Dataset found")
        logger.info("✓ All prerequisites met\n")
        return True
    
    def run_scenario(self, scenario_name, server_script, num_honest=2):
        """Run a single experiment scenario."""
        logger.info("\n" + "="*60)
        logger.info(f"SCENARIO: {scenario_name}")
        logger.info("="*60)
        
        processes = []
        
        try:
            # Start server
            logger.info(f"\n1. Starting {scenario_name} server...")
            server_path = os.path.join("federated", server_script)
            server_proc = subprocess.Popen(
                [sys.executable, server_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            processes.append(("Server", server_proc))
            time.sleep(5)  # Give server time to start
            logger.info("✓ Server started")
            
            # Start honest clients
            logger.info(f"\n2. Starting {num_honest} honest clients...")
            for i in range(num_honest):
                # Modify CLIENT_ID and VIDEO_SOURCE for each client
                # For simplicity, we'll just run the default script
                # In production, you'd pass arguments or use different configs
                client_proc = subprocess.Popen(
                    [sys.executable, os.path.join("federated", "federated_client.py")],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                processes.append((f"Client-{i+1}", client_proc))
                time.sleep(2)
            logger.info(f"✓ {num_honest} honest clients started")
            
            # Start malicious client
            logger.info("\n3. Starting malicious client...")
            attacker_proc = subprocess.Popen(
                [sys.executable, os.path.join("federated", "malicious_client.py")],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            processes.append(("Attacker", attacker_proc))
            logger.info("✓ Malicious client started")
            
            # Wait for experiment to complete
            logger.info(f"\n4. Running experiment for {NUM_FEDERATED_ROUNDS} rounds...")
            logger.info("   (This may take several minutes)")
            
            # Wait for server to finish
            server_proc.wait(timeout=600)  # 10 minute timeout
            
            logger.info("✓ Server completed")
            
            # Terminate all clients
            logger.info("\n5. Stopping all clients...")
            for name, proc in processes[1:]:  # Skip server
                proc.terminate()
                proc.wait(timeout=5)
            logger.info("✓ All clients stopped")
            
            logger.info(f"\n✅ {scenario_name} scenario complete!")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("⏱️  Experiment timed out!")
            return False
        except Exception as e:
            logger.error(f"❌ Error during experiment: {str(e)}")
            return False
        finally:
            # Cleanup: terminate all processes
            for name, proc in processes:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    proc.kill()
    
    def run_full_experiment(self):
        """Run complete experiment: both scenarios + evaluation."""
        start_time = time.time()
        
        logger.info("\n" + "="*60)
        logger.info("STARTING FULL FEDGUARD EXPERIMENT")
        logger.info("="*60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites not met. Aborting.")
            return False
        
        # Scenario 1: Secure (with FedGuard)
        logger.info("\n📊 SCENARIO 1: FEDGUARD (SECURE)")
        success_secure = self.run_scenario(
            "Secure", 
            "fedguard_server.py",
            num_honest=2
        )
        
        if not success_secure:
            logger.error("Secure scenario failed. Aborting.")
            return False
        
        time.sleep(5)  # Brief pause between scenarios
        
        # Scenario 2: Insecure (no defense)
        logger.info("\n📊 SCENARIO 2: NO DEFENSE (INSECURE)")
        success_insecure = self.run_scenario(
            "Insecure",
            "insecure_server.py",
            num_honest=2
        )
        
        if not success_insecure:
            logger.error("Insecure scenario failed.")
            return False
        
        # Evaluation
        logger.info("\n📊 RUNNING EVALUATION...")
        eval_proc = subprocess.run(
            [sys.executable, os.path.join("evaluation", "evaluate_model.py")],
            capture_output=True,
            text=True
        )
        
        if eval_proc.returncode != 0:
            logger.error("Evaluation failed!")
            logger.error(eval_proc.stderr)
            return False
        
        logger.info("✓ Evaluation complete")
        
        # Visualization
        logger.info("\n📊 GENERATING VISUALIZATIONS...")
        viz_proc = subprocess.run(
            [sys.executable, os.path.join("evaluation", "visualize_results.py")],
            capture_output=True,
            text=True
        )
        
        if viz_proc.returncode != 0:
            logger.error("Visualization failed!")
            logger.error(viz_proc.stderr)
            return False
        
        logger.info("✓ Visualizations generated")
        
        # Summary
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("✅ EXPERIMENT COMPLETE!")
        logger.info("="*60)
        logger.info(f"Total duration: {duration/60:.2f} minutes")
        logger.info(f"Results saved in: {RESULTS_DIR}")
        logger.info("\nGenerated files:")
        logger.info(f"  • Models: {MODELS_DIR}")
        logger.info(f"  • Metrics: {METRICS_DIR}")
        logger.info(f"  • Plots: {PLOTS_DIR}")
        logger.info(f"  • Logs: {LOGS_DIR}")
        
        return True


def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║           FEDGUARD AUTOMATED EXPERIMENT RUNNER            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
          This script will:
1. Run FedGuard secure server with 2 honest + 1 malicious client
2. Run insecure server with 2 honest + 1 malicious client
3. Evaluate both models
4. Generate comprehensive visualizations

Prerequisites:
- Baseline model trained (run train_baseline_model.py)
- Kafka server running
- Dataset downloaded

""")

response = input("Ready to start? (y/n): ")
if response.lower() != 'y':
    print("Experiment cancelled.")
runner = ExperimentRunner("fedguard_full_experiment")
success = runner.run_full_experiment()

if success:
    print("\n🎉 Experiment completed successfully!")
    print(f"Check {RESULTS_DIR} for all outputs.")
else:
    print("\n❌ Experiment failed. Check logs for details.")