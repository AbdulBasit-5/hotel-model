# Hotel Booking Cancellation ML & Research Platform

This project provides an interactive **Angular 22 frontend** and **Flask REST API backend** designed to showcase, benchmark, and simulate the machine learning models trained on the **Hotel Booking Demand** dataset (*Antonio et al., 2019*), reproducing and critically examining published literature claims.

---

## 🔬 Research & Code Analysis Summary

### 1. The Dataset & Research Goal
* **Total Records**: 119,390 bookings (79,330 City Hotel, 40,060 Resort Hotel).
* **Target**: `is_canceled` (Overall cancellation rate: **37.0%**; City Hotel: **41.7%**, Resort Hotel: **27.8%**).
* **Research Goal**: Test whether published tabular deep learning models (**TabNet**, which claimed $\approx 0.862$ AUC-ROC on Resort and $\approx 0.846$ on City) actually outperform standard tree-based models when evaluated under honest protocols.

### 2. Critical Findings in Your Code
1. **Severe Target Leakage Identified**:
   * Features `reservation_status` (Check-Out, Canceled, No-Show) and `reservation_status_date` directly leak the label with 100% certainty. Your code correctly dropped them (`ALWAYS_DROP`).
   * Operational post-booking columns (`assigned_room_type`, `booking_changes`, `days_in_waiting_list`) were flagged as `RISKY` and tested via an ablation study.
2. **Duplicate Record Leakage (The 26.8% Discovery)**:
   * **31,994 exact duplicate rows** were discovered.
   * Standard random train/test splits allow duplicates to exist in both training and test sets, artificially inflating metrics.
3. **The 3 Evaluation Protocols**:
   * **Protocol 1 (P1 - Random, Raw)**: Duplicates present. Naive random 80/20 split. Scores are inflated ($\text{AUC-ROC} \approx 0.96$).
   * **Protocol 2 (P2 - Random, Dedup)**: Duplicates removed, random split. Scores adjust to $\approx 0.94$ (Resort) and $\approx 0.83$ (City).
   * **Protocol 3 (P3 - Time-Based, Dedup)**: Strict chronological split on deduplicated data (trains on bookings up to Feb 2017, tests on future arrivals). Mirrors real-world production performance ($\text{AUC-ROC} \approx 0.88$ for Resort, $\approx 0.83$ for City).
4. **Gate 1 Check (TabNet vs GBDTs)**:
   * Standard GBDTs (**XGBoost**, **LightGBM**, **CatBoost**, **RandomForest**) consistently outperform TabNet while training in seconds rather than requiring GPU deep learning training loops.

---

## 🛠️ Architecture & Implementation

### 1. Backend (`backend/`) - Flask API
* **`app.py`**:
  * `GET /api/health`: Health status and pipeline metrics.
  * `GET /api/overview`: Dataset dimensions, duplicate leakage stats, protocols.
  * `GET /api/benchmarks`: Benchmark matrix of 5 models + TabNet across all 3 protocols and both hotels with all 7 metrics (AUC-ROC, AUC-PR, F1, Accuracy, Precision, Recall, Brier score).
  * `GET /api/tabnet-comparison`: Gate 1 comparison table.
  * `GET /api/ablation`: Impact of dropping risky columns under Protocol 3.
  * `POST /api/predict`: Live inference engine calculating real-time cancellation probability, risk level (`LOW`, `MEDIUM`, `HIGH`), and factor impact breakdown.
* **`train_model.py`**:
  * Trains a high-performance `HistGradientBoostingClassifier` pipeline (AUC-ROC: **0.9484**, Accuracy: **87.57%**, F1: **0.8249**).
  * Serializes model to `backend/model_pipeline.joblib`.

### 2. Frontend (`frontend/`) - Angular 22
* Built with Angular standalone components, signals, and a modern dark-mode aesthetic.
* **Tabs**:
  1. **Executive Overview & Leakage**: High-level KPIs, hotel distributions, leakage remediation analysis, protocol definitions.
  2. **Model Benchmarks (5 Models × 3 Protocols)**: Interactive filters for Hotel (H1 / H2), Protocol (P1 / P2 / P3), and Sort Metric (AUC-ROC, F1, Accuracy, etc.) with animated comparative bar charts and summary tables.
  3. **Gate 1 Check & TabNet Paper**: Comparison of published TabNet vs our reproduced models, plus risky feature ablation table.
  4. **Live Prediction Simulator**: Interactive reservation form with pre-configured presets (High-Risk OTA, Safe Family Vacation, Corporate Traveler), a circular risk gauge, probability meter, and actionable hotel revenue management recommendations.

---

## 🚀 How to Run the Application

### Option A: One-Click Launch (Windows)
Double-click `start_servers.bat` in the project root.

### Option B: Manual Command Line
1. **Start the Flask Backend**:
   ```bash
   python backend/app.py
   ```
   *Runs on `http://127.0.0.1:5000`*

2. **Start the Angular Frontend**:
   ```bash
   cd frontend
   npm start
   ```
   *Runs on `http://localhost:4200`*

3. Open your browser and navigate to **`http://localhost:4200`**.
