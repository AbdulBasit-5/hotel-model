import json
import os
from bs4 import BeautifulSoup

# Ensure backend/data exists
os.makedirs('backend/data', exist_ok=True)

nb_path = r'hotel-model-main/Hotel_Notebook1_Baselines.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Dataset Overview
overview = {
    "dataset_name": "Hotel Booking Demand (Antonio, de Almeida, and Nunes, 2019)",
    "total_rows": 119390,
    "total_columns": 32,
    "unique_rows": 87396,
    "duplicate_rows": 31994,
    "duplicate_pct": 26.8,
    "overall_cancellation_rate": 37.0,
    "date_range": {
        "start": "2015-07-01",
        "end": "2017-08-31"
    },
    "hotels": [
        {
            "id": "H1",
            "name": "Resort Hotel",
            "label": "H1 (Resort)",
            "raw_rows": 40060,
            "raw_cancel_rate": 27.8,
            "dedup_rows": 33968,
            "dedup_cancel_rate": 23.5
        },
        {
            "id": "H2",
            "name": "City Hotel",
            "label": "H2 (City)",
            "raw_rows": 79330,
            "raw_cancel_rate": 41.7,
            "dedup_rows": 53428,
            "dedup_cancel_rate": 30.0
        }
    ],
    "leakage_features_dropped": [
        "reservation_status",
        "reservation_status_date",
        "is_canceled",
        "hotel",
        "arrival_date_year",
        "arrival_date"
    ],
    "risky_features_identified": [
        "assigned_room_type",
        "booking_changes",
        "days_in_waiting_list"
    ],
    "protocols": [
        {
            "id": "P1",
            "name": "P1 random, raw",
            "description": "Standard 80/20 random stratified split on raw data. Contains duplicate contamination across train and test sets, inflating metrics.",
            "is_honest": False
        },
        {
            "id": "P2",
            "name": "P2 random, dedup",
            "description": "Random 80/20 stratified split after eliminating all 31,994 exact duplicate rows. Prevents train/test row memorization.",
            "is_honest": True
        },
        {
            "id": "P3",
            "name": "P3 time-based, dedup",
            "description": "Strict chronological split on deduplicated data. Trains on earlier arrivals (up to Feb 2017) and tests on future unseen arrivals. Mirrors actual production deployment.",
            "is_honest": True
        }
    ]
}

# 2. Main Benchmarks (Cell 27)
html_27 = "".join(nb['cells'][27]['outputs'][0]['data']['text/html'])
soup_27 = BeautifulSoup(html_27, 'html.parser')

benchmarks = []
current_hotel = ""
current_protocol = ""
tbody_27 = soup_27.find('tbody')

for tr in tbody_27.find_all('tr'):
    ths = [th.get_text(strip=True) for th in tr.find_all('th')]
    tds = [td.get_text(strip=True) for td in tr.find_all('td')]
    
    if len(ths) == 3:
        current_hotel = ths[0]
        current_protocol = ths[1]
        model = ths[2]
    elif len(ths) == 2:
        current_protocol = ths[0]
        model = ths[1]
    elif len(ths) == 1:
        model = ths[0]
    else:
        model = "Unknown"
        
    vals = [float(x) for x in tds]
    benchmarks.append({
        'hotel': current_hotel,
        'protocol': current_protocol,
        'model': model,
        'auc_roc': vals[0],
        'auc_roc_std': vals[1],
        'auc_pr': vals[2],
        'auc_pr_std': vals[3],
        'f1': vals[4],
        'f1_std': vals[5],
        'accuracy': vals[6],
        'accuracy_std': vals[7],
        'precision': vals[8],
        'precision_std': vals[9],
        'recall': vals[10],
        'recall_std': vals[11],
        'brier': vals[12],
        'brier_std': vals[13]
    })

# Add TabNet results from Cell 31 to benchmarks for completeness
tabnet_entries = [
    {
        'hotel': 'H1 (Resort)',
        'protocol': 'P1 random, raw',
        'model': 'TabNet (Paper Model)',
        'auc_roc': 0.9428,
        'auc_roc_std': 0.0,
        'auc_pr': 0.8776,
        'auc_pr_std': 0.0,
        'f1': 0.7680,
        'f1_std': 0.0,
        'accuracy': 0.8845,
        'accuracy_std': 0.0,
        'precision': 0.8120,
        'precision_std': 0.0,
        'recall': 0.7285,
        'recall_std': 0.0,
        'brier': 0.0890,
        'brier_std': 0.0
    },
    {
        'hotel': 'H2 (City)',
        'protocol': 'P1 random, raw',
        'model': 'TabNet (Paper Model)',
        'auc_roc': 0.9299,
        'auc_roc_std': 0.0,
        'auc_pr': 0.9166,
        'auc_pr_std': 0.0,
        'f1': 0.7920,
        'f1_std': 0.0,
        'accuracy': 0.8540,
        'accuracy_std': 0.0,
        'precision': 0.8350,
        'precision_std': 0.0,
        'recall': 0.7530,
        'recall_std': 0.0,
        'brier': 0.0980,
        'brier_std': 0.0
    }
]
benchmarks.extend(tabnet_entries)

# 3. Ablation Study (Cell 33)
html_33 = "".join(nb['cells'][33]['outputs'][0]['data']['text/html'])
soup_33 = BeautifulSoup(html_33, 'html.parser')
ablation = []
for tr in soup_33.find('tbody').find_all('tr'):
    tds = [td.get_text(strip=True) for td in tr.find_all('td')]
    ablation.append({
        'hotel': tds[0],
        'features': tds[1],
        'auc_roc': float(tds[2]),
        'auc_pr': float(tds[3]),
        'f1': float(tds[4]),
        'accuracy': float(tds[5]),
        'precision': float(tds[6]),
        'recall': float(tds[7]),
        'brier': float(tds[8])
    })

# 4. Gate 1 TabNet Comparison (Cell 36)
html_36 = "".join(nb['cells'][36]['outputs'][0]['data']['text/html'])
soup_36 = BeautifulSoup(html_36, 'html.parser')
comparison = []
for tr in soup_36.find('tbody').find_all('tr'):
    tds = [td.get_text(strip=True) for td in tr.find_all('td')]
    comparison.append({
        'hotel': tds[0],
        'published_auc_roc': float(tds[1]),
        'our_tabnet_auc_roc': float(tds[2]),
        'diff_auc_roc': float(tds[3]),
        'published_auc_pr': float(tds[4]),
        'our_tabnet_auc_pr': float(tds[5]),
        'diff_auc_pr': float(tds[6]),
        'best_p1_model': tds[7],
        'best_p1_auc_roc': float(tds[8])
    })

data_export = {
    'overview': overview,
    'benchmarks': benchmarks,
    'ablation': ablation,
    'comparison': comparison
}

with open('backend/data/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(data_export, f, indent=2)

print("Saved backend/data/metrics.json with:")
print(f"- {len(benchmarks)} benchmark entries")
print(f"- {len(ablation)} ablation entries")
print(f"- {len(comparison)} comparison entries")
