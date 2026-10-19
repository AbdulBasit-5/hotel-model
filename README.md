# Hotel Booking Cancellation Prediction & Research Platform

An academic research platform, interactive **Angular 22 dashboard**, and **Flask REST API** evaluating machine learning models on the Antonio et al. (*Hotel Booking Demand*) dataset (119,390 records).

---

## 🔬 Research Context & Integrity

### 1. Dataset Overview
* **Source**: Antonio, de Almeida, and Nunes (2019) Hotel Booking Demand dataset.
* **Volume**: 119,390 total bookings spanning July 2015 to August 2017:
  * **City Hotel (H2)**: 79,330 bookings (41.7% raw cancellation rate)
  * **Resort Hotel (H1)**: 40,060 bookings (27.8% raw cancellation rate)
* **Target Variable**: `is_canceled` (0 = Check-Out / Stayed, 1 = Canceled). Overall baseline cancellation rate: **37.0%**.

### 2. Critical Findings & Data Leakage Remediation
* **Target Leakage**: Columns `reservation_status` (Check-Out, Canceled, No-Show) and `reservation_status_date` directly reveal the target outcome with 100% certainty. These are strictly excluded (`ALWAYS_DROP`).
* **The 26.8% Duplicate Contamination Finding**:
  * **31,994 exact duplicate rows** exist in the raw dataset (25,902 in City Hotel, 6,092 in Resort Hotel).
  * In naive random splits, duplicates appear across both train and test partitions, resulting in row memorization and inflated performance scores.
* **Operational Feature Ablation**:
  * Post-booking variables (`assigned_room_type`, `booking_changes`, `days_in_waiting_list`) were isolated as `RISKY` and tested via an ablation study to verify whether their removal degrades generalization.

### 3. The Three Evaluation Protocols
* **Protocol 1 (P1 - Random Split, Raw Data)**: Standard 80/20 random stratified split on raw data with duplicates present. Matches literature conventions but yields overly optimistic metrics ($\text{AUC-ROC} \approx 0.959$–$0.962$) due to duplicate leakage.
* **Protocol 2 (P2 - Random Split, Deduplicated Data)**: All 31,994 exact duplicates removed before an 80/20 stratified split. AUC-ROC adjusts to $\approx 0.942$ (Resort) and $\approx 0.830$ (City).
* **Protocol 3 (P3 - Time-Based Split, Deduplicated Data)**: Strict chronological split on deduplicated data. Models train on past bookings (up to Feb 2017) and are evaluated strictly on future unseen arrivals. Represents genuine real-world production performance ($\text{AUC-ROC} \approx 0.888$ for Resort, $\approx 0.837$ for City; F1 drops to $0.55$–$0.62$).

### 4. Gate 1 Academic Check: TabNet vs. GBDTs
* TabNet (tabular deep learning) was reproduced to verify literature claims (published AUC-ROC $\approx 0.862$ on Resort, $\approx 0.846$ on City).
* Our reproduced TabNet under P1 reached $0.9428$ (Resort) and $0.9299$ (City), showing that published differences stem from differing preprocessing or hyperparameters.
* Gradient-boosted decision trees (**CatBoost**, **XGBoost**, **LightGBM**, **Random Forest**) consistently outperform TabNet while training in seconds rather than requiring GPU training loops.

### 5. Project Scope & Stage 2 Limitation
* **Stage 1 (Current Implementation)**: Binary classification predicting whether a booking will cancel (`is_canceled`).
* **Stage 2 (Future Work)**: Predicting duration until cancellation or time-to-event via survival analysis models. The current classifier does **not** predict a cancellation date, hazard curve, or remaining days to cancellation.
* **Deployed Interactive Model**: The live simulator utilizes a `HistGradientBoostingClassifier` trained on an 80/20 random stratified split of the raw data for fast interactive demonstration. For genuine production generalization, refer to Protocol 3 benchmarks.

---

## 🏛️ System Architecture

```
hotel-model/
├── .github/workflows/
│   ├── ci.yml                 # Automated CI: runs backend tests and builds frontend
│   └── pages.yml              # GitHub Pages deployment workflow for Angular frontend
├── backend/
│   ├── app.py                 # Hardened Flask REST API (CORS, validation, batch inference)
│   ├── requirements.txt       # Pinned backend dependencies
│   ├── train_model.py         # Pipeline training script
│   ├── model_pipeline.joblib  # Serialized scikit-learn pipeline
│   ├── model_metadata.json    # Pipeline metadata and training parameters
│   ├── data/
│   │   └── metrics.json       # Extracted benchmark, TabNet, and ablation metrics
│   └── tests/
│       ├── __init__.py
│       └── test_api.py        # Automated backend test suite (14 test cases)
├── frontend/
│   ├── src/
│   │   ├── app/               # Angular 22 standalone components
│   │   │   ├── app.ts         # Logic, signals, and API integration
│   │   │   ├── app.html       # 4 research views with accessible controls
│   │   │   ├── app.css        # Component stylesheet
│   │   │   └── app.spec.ts    # Frontend unit test suite (6 test cases)
│   │   ├── environments/      # Environment configurations (dev vs prod)
│   │   └── styles.css         # Global design system
│   ├── public/data/
│   │   └── metrics.json       # Bundled static research data for offline/Pages usage
│   ├── angular.json           # Angular CLI workspace configuration
│   └── package.json           # Frontend dependencies & npm scripts
├── start_servers.bat          # One-click Windows development startup script
├── .gitignore                 # Excludes caches, node_modules, and raw datasets
└── README.md                  # Project documentation
```

---

## 💻 Local Development Setup

### Prerequisites
* **Python**: 3.10, 3.11, 3.12, 3.13, or 3.14
* **Node.js**: 20 LTS or 22 LTS
* **npm**: 10 or 11

### 1. One-Click Startup (Windows)
Double-click `start_servers.bat` in the root folder. It starts the Flask backend on port 5000 and the Angular dev server on port 4200.

### 2. Manual Startup

#### Step A: Flask Backend
```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Run backend tests
python -m pytest backend/tests -v

# 3. Start Flask API
python backend/app.py
```
*API runs at `http://127.0.0.1:5000`*

#### Step B: Angular Frontend
```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies reproducibly
npm ci

# 3. Run unit tests
npx ng test --watch=false

# 4. Start Angular development server
npm start
```
*Frontend runs at `http://localhost:4200`*

---

## 🧪 Testing Commands

### Backend Tests
```bash
python -m pytest backend/tests -v
```
Runs 14 test cases covering:
* Health check schema and Stage 1 labeling
* Dataset overview and model listing
* Protocol and hotel benchmark filtering
* Valid single and batch predictions
* Validation for negative lead times, invalid ADR, and unsupported categories
* 503 error handling when the model pipeline is missing (preventing fake predictions)

### Frontend Tests
```bash
cd frontend
npx ng test --watch=false
```
Runs 6 unit tests covering component initialization, title rendering, tab navigation, benchmark filtering, preset switching, and offline API error states.

### Frontend Production Build
```bash
cd frontend
npm run build
```
Generates the optimized static distribution inside `frontend/dist/frontend/browser`.

---

## 🌐 GitHub Pages Deployment & Actions

### 1. Workflows Included
* **`.github/workflows/ci.yml`**:
  * Triggered on pushes and pull requests to `main`.
  * Runs backend test suite via pytest.
  * Runs Angular unit tests and validates production compilation.
* **`.github/workflows/pages.yml`**:
  * Triggered on pushes to `main` impacting frontend or manual dispatch.
  * Builds Angular with `--base-href "/${{ github.event.repository.name }}/"`.
  * Deploys static build artifact (`frontend/dist/frontend/browser`) directly to GitHub Pages.

### 2. Setting Up GitHub Pages
1. Push the repository to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete research dashboard with CI and GitHub Pages deployment"
   git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
   git push -u origin main
   ```
2. Navigate to your repository on GitHub: **Settings** → **Pages**.
3. Under **Build and deployment** → **Source**, select **GitHub Actions**.
4. The deployment workflow will trigger automatically. Inspect progress under the **Actions** tab.
5. Your dashboard will be live at `https://<YOUR_USERNAME>.github.io/<YOUR_REPOSITORY>/`.

### 3. API Separation & Public Site Behavior
* **GitHub Pages hosts static files only** and cannot execute Python or Flask servers.
* **What works out-of-the-box on GitHub Pages**:
  * All research dashboards (Executive Overview, Leakage Analysis, Benchmark Matrix, Gate 1 Check, Ablation Study) function **100%** using the bundled research data (`data/metrics.json`).
* **Live Prediction Simulator on GitHub Pages**:
  * If the Flask backend is not deployed publicly, the simulator displays a clear message stating that live inference requires an active backend. It does **not** generate fake or simulated predictions.
  * To enable live predictions on the public site:
    1. Deploy `backend/` to a cloud service (e.g., Render, Railway, AWS, Fly.io).
    2. Configure the production API URL in `frontend/src/environments/environment.prod.ts` or via `window.__APP_CONFIG__ = { apiUrl: 'https://your-api.onrender.com' };`.
    3. Ensure CORS is configured in `backend/app.py` to allow requests from your GitHub Pages domain.

---

## 📋 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health status, loaded model name, validation metrics, and stage notices |
| `GET` | `/api/overview` | Total bookings, cancellation baseline, duplicate count, protocols |
| `GET` | `/api/models` | Metadata for deployed model and evaluated benchmark algorithms |
| `GET` | `/api/benchmarks` | Full evaluation matrix (filters: `?hotel=H1` and `?protocol=P3`) |
| `GET` | `/api/metrics` | Alias for `/api/benchmarks` |
| `GET` | `/api/tabnet-comparison`| Published TabNet results vs. reproduced tree baselines |
| `GET` | `/api/ablation` | Impact of removing risky post-booking columns under Protocol 3 |
| `GET` | `/api/feature-defaults` | Default baseline values and valid categorical options |
| `POST` | `/api/predict` | Single reservation inference returning probability, risk tier, and heuristic indicators |
| `POST` | `/api/predict/batch` | Vectorized batch inference for up to 100 reservations |

---

## ⚖️ License & Citations
* **Dataset**: Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41-49.
* **Paper Comparison**: Evaluated against published TabNet implementations on hotel booking cancellation benchmarks.
>>>>>>> 83b8e3b (feat: complete hotel cancellation ML platform with Flask API, Angular 22 dashboard, CI and GitHub Pages deployment)
