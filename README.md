

---

# FedGuard: A Unified Defense Framework for Federated Learning

**FedGuard** is a dual-layer security framework designed to protect Federated Learning (FL) systems against simultaneous **Model Poisoning (Integrity)** and **DDoS (Availability)** attacks. This project demonstrates a robust defense architecture applied to **Video Anomaly Detection** using the UCSD Pedestrian Dataset.

## 👥 Project Team

**Department of Computer Science & Engineering, SRM University-AP**

* **N. Nandini Devi** (AP22110010593)
* **A. Sri Lakshmi** (AP22110010614)
* **G. Sri Vyshnavi** (AP22110010632)
* **D. Nandini** (AP22110010585)

**Guided By:** Dr. SriRamulu Bojjagani

---

## 🚀 Key Features

FedGuard bridges the gap between network security and algorithmic robustness:

1. **Dual-Layer Defense Architecture**:
* **Layer 1 (Availability):** Prevents Distributed Denial of Service (DDoS) attacks using **Apache Kafka** and timestamp-based dynamic rate limiting.
* **Layer 2 (Integrity):** Filters malicious model updates using a two-stage process:
* *Performance Validation:* Checks reconstruction loss on a trusted server-side root dataset (32 frames).
* *Statistical Consensus:* Uses **Median Absolute Deviation (MAD)** to detect and reject subtle statistical outliers.




2. **Robustness**:
* Maintains **94.2% Accuracy** under poisoning attacks (vs 23.7% for unprotected systems).
* Reduces Server CPU load from **100% to 40%** during DDoS flooding.


3. **Application Domain**: Video Anomaly Detection using Convolutional Autoencoders (CAE).

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** TensorFlow 2.16, Keras, NumPy, OpenCV
* **Communication:** Apache Kafka, Zookeeper
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure

```bash
FedGuard/
├── federated/
│   ├── fedguard_server.py       # Main Secure Server (Defense Logic)
│   ├── insecure_server.py       # Baseline Server (For comparison)
│   ├── federated_client.py      # Honest FL Client
│   ├── malicious_client.py      # Poisoning Attack Client
│   └── malicious_client_ddos.py # DDoS Attack Bot
├── utils/
│   ├── security.py              # MAD & Loss Verification Algorithms
│   └── data_loader.py           # Efficient Video Frame Generator
├── config.py                    # Global Configuration (Thresholds, Paths)
├── requirements.txt             # Python Dependencies
└── README.md

```

---

## ⚙️ Installation & Setup

### 1. Prerequisites

Ensure you have **Python** and **Java** (for Kafka) installed.

* **Kafka Setup:** Download and extract Apache Kafka.
```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties
# Start Kafka Server
bin/kafka-server-start.sh config/server.properties

```



### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Configure Dataset

1. Download the **UCSD Pedestrian Dataset (Ped1)**.
2. Open `config.py` and set `DATASET_ROOT` to your local dataset path.

---

## 🏃 Usage

### 1. Start the FedGuard Server

Run the secure server which initializes the Kafka consumer and loads the initial global model.

```bash
python federated/fedguard_server.py

```

### 2. Start Honest Clients

Launch legitimate clients to begin training.

```bash
python federated/federated_client.py

```

### 3. Simulate Attacks (Optional)

To test the defenses, launch adversarial clients in separate terminals:

* **Launch Poisoning Attack (Integrity):**
```bash
python federated/malicious_client.py

```


* **Launch DDoS Attack (Availability):**
```bash
python federated/malicious_client_ddos.py

```



---

## 📊 Results

| Metric | Unprotected System | FedGuard System | Improvement |
| --- | --- | --- | --- |
| **Model Accuracy** | 23.7% (Random) | **94.2% (High Precision)** | +70.5% |
| **Reconstruction Error** | ~0.32 (High) | **~0.003 (Low)** | 99% Reduction |
| **CPU Load (DDoS)** | 100% (Crash) | **40% (Stable)** | 60% Reduction |
| **Attack Detection** | 0% | **100%** | Critical |

---

## 🔮 Future Scope

* **Homomorphic Encryption:** To validate updates without decrypting raw weights.
* **Blockchain Integration:** To maintain an immutable reputation score for clients.
* **Adaptive Thresholding:** Dynamic adjustment of rate limits based on network traffic analysis.
