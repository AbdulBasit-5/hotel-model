import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Restrict CORS to local development and standard GitHub Pages / production origins
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Path to research data artifacts
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
METRICS_PATH = os.path.join(DATA_DIR, 'metrics.json')

METRICS_DATA = {}
try:
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        METRICS_DATA = json.load(f)
except Exception as e:
    print(f"Warning: Could not load metrics.json from {METRICS_PATH}: {e}")

# Controlled artifact path for trained model pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_pipeline.joblib')
METADATA_PATH = os.path.join(os.path.dirname(__file__), 'model_metadata.json')

pipeline = None
metadata = {}

def load_model_pipeline(path=MODEL_PATH, meta_path=METADATA_PATH):
    """Safely load trained pipeline and metadata from trusted local filesystem path."""
    global pipeline, metadata
    if not os.path.isfile(path):
        print(f"Notice: Model artifact not found at {path}. Live predictions will return 503 until trained.")
        pipeline = None
        metadata = {}
        return False
    try:
        pipeline = joblib.load(path)
        if os.path.isfile(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        print(f"Model pipeline successfully loaded from {path}")
        return True
    except Exception as e:
        print(f"Error loading model pipeline from {path}: {e}")
        pipeline = None
        metadata = {}
        return False

# Initial load
load_model_pipeline()

VALID_OPTIONS = {
    'hotel': ['City Hotel', 'Resort Hotel'],
    'arrival_date_month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'meal': ['BB', 'HB', 'FB', 'SC', 'Undefined'],
    'market_segment': ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate', 'Groups', 'Complementary', 'Aviation'],
    'distribution_channel': ['TA/TO', 'Direct', 'Corporate', 'GDS', 'Undefined'],
    'deposit_type': ['No Deposit', 'Non Refund', 'Refundable'],
    'customer_type': ['Transient', 'Transient-Party', 'Contract', 'Group'],
    'reserved_room_type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'P'],
    'assigned_room_type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'P']
}

DEFAULT_INPUTS = {
    'hotel': 'City Hotel',
    'lead_time': 90,
    'arrival_date_month': 'July',
    'arrival_date_week_number': 27,
    'arrival_date_day_of_month': 5,
    'stays_in_weekend_nights': 2,
    'stays_in_week_nights': 3,
    'adults': 2,
    'children': 0,
    'babies': 0,
    'meal': 'BB',
    'country': 'PRT',
    'market_segment': 'Online TA',
    'distribution_channel': 'TA/TO',
    'is_repeated_guest': 0,
    'previous_cancellations': 0,
    'previous_bookings_not_canceled': 0,
    'reserved_room_type': 'A',
    'assigned_room_type': 'A',
    'booking_changes': 0,
    'deposit_type': 'No Deposit',
    'agent': '9',
    'company': '-1',
    'days_in_waiting_list': 0,
    'customer_type': 'Transient',
    'adr': 105.0,
    'required_car_parking_spaces': 0,
    'total_of_special_requests': 1
}

def validate_booking_input(data):
    """Validate user input fields, types, and ranges. Returns (validated_dict, error_message)."""
    if not isinstance(data, dict):
        return None, "Input payload must be a JSON object"

    # Merge with default baseline values
    merged = {**DEFAULT_INPUTS, **data}

    # Categorical validations
    for cat_field, allowed in VALID_OPTIONS.items():
        if cat_field in merged:
            val = merged[cat_field]
            if val not in allowed:
                return None, f"Invalid value '{val}' for field '{cat_field}'. Allowed: {allowed}"

    # Numeric range validations
    numeric_constraints = {
        'lead_time': (0, 1000, int),
        'adr': (0.0, 10000.0, float),
        'adults': (0, 50, int),
        'children': (0, 20, int),
        'babies': (0, 20, int),
        'stays_in_weekend_nights': (0, 50, int),
        'stays_in_week_nights': (0, 100, int),
        'is_repeated_guest': (0, 1, int),
        'previous_cancellations': (0, 50, int),
        'previous_bookings_not_canceled': (0, 100, int),
        'booking_changes': (0, 50, int),
        'days_in_waiting_list': (0, 500, int),
        'required_car_parking_spaces': (0, 20, int),
        'total_of_special_requests': (0, 10, int)
    }

    for num_field, (min_v, max_v, cast_type) in numeric_constraints.items():
        if num_field in merged:
            try:
                raw_val = merged[num_field]
                val = cast_type(raw_val)
                if val < min_v or val > max_v:
                    return None, f"Field '{num_field}' value {val} out of bounds [{min_v}, {max_v}]"
                merged[num_field] = val
            except (ValueError, TypeError):
                return None, f"Field '{num_field}' must be a valid {cast_type.__name__}"

    # IDs as string
    merged['agent'] = str(merged.get('agent', '-1'))
    merged['company'] = str(merged.get('company', '-1'))
    merged['country'] = str(merged.get('country', 'PRT'))

    return merged, None

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': pipeline is not None,
        'model_name': 'HistGradientBoostingClassifier',
        'stage': 'Stage 1: Binary Cancellation Prediction',
        'stage_2_status': 'Not implemented (Future Work: Time-to-Cancellation / Survival Analysis)',
        'model_accuracy': metadata.get('metrics', {}).get('accuracy', 0.8757 if pipeline else None),
        'model_auc_roc': metadata.get('metrics', {}).get('auc_roc', 0.9484 if pipeline else None),
        'evaluation_protocol': 'P1-style 80/20 random stratified split on raw data (for live interactive demonstration)',
        'research_protocol_note': 'For authentic real-world generalization without duplicate leakage, refer to Protocol 3 benchmarks.'
    })

@app.route('/api/overview', methods=['GET'])
def get_overview():
    if 'overview' not in METRICS_DATA:
        return jsonify({'error': 'Research overview data unavailable'}), 404
    return jsonify(METRICS_DATA['overview'])

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'deployed_model': {
            'name': 'HistGradientBoostingClassifier (Scikit-Learn)',
            'preprocessing': 'ColumnTransformer(OrdinalEncoder for categories + Median Imputer for numericals)',
            'evaluation_protocol': 'P1-style Random Stratified Split (Raw Data)',
            'metrics': metadata.get('metrics', {'auc_roc': 0.9484, 'accuracy': 0.8757, 'f1': 0.8249}),
            'active': pipeline is not None
        },
        'benchmark_models_evaluated': [
            'LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'TabNet'
        ],
        'protocols_evaluated': ['P1 random, raw', 'P2 random, dedup', 'P3 time-based, dedup']
    })

@app.route('/api/benchmarks', methods=['GET'])
@app.route('/api/metrics', methods=['GET'])
def get_benchmarks():
    if 'benchmarks' not in METRICS_DATA:
        return jsonify({'error': 'Benchmark research data unavailable'}), 404

    hotel_filter = request.args.get('hotel')
    protocol_filter = request.args.get('protocol')
    
    rows = METRICS_DATA['benchmarks']
    if hotel_filter:
        rows = [r for r in rows if hotel_filter.lower() in r['hotel'].lower()]
    if protocol_filter:
        rows = [r for r in rows if protocol_filter.lower() in r['protocol'].lower()]
        
    return jsonify({
        'count': len(rows),
        'benchmarks': rows
    })

@app.route('/api/tabnet-comparison', methods=['GET'])
def get_tabnet_comparison():
    if 'comparison' not in METRICS_DATA:
        return jsonify({'error': 'TabNet comparison data unavailable'}), 404
    return jsonify({
        'description': "Gate 1 Check: Reproduction of Published TabNet Results vs Our Best Tree Baselines",
        'note': "Published paper metrics reported from literature. Our TabNet reproduced under P1 protocol.",
        'comparisons': METRICS_DATA['comparison']
    })

@app.route('/api/ablation', methods=['GET'])
def get_ablation():
    if 'ablation' not in METRICS_DATA:
        return jsonify({'error': 'Ablation study data unavailable'}), 404
    return jsonify({
        'description': "Ablation Study: Protocol 3 (Time-based, dedup) with and without risky columns (assigned_room_type, booking_changes, days_in_waiting_list)",
        'ablation': METRICS_DATA['ablation']
    })

@app.route('/api/feature-defaults', methods=['GET'])
def get_feature_defaults():
    return jsonify({
        'defaults': DEFAULT_INPUTS,
        'options': VALID_OPTIONS
    })

def compute_heuristic_factors(features):
    """
    Derives heuristic risk indicators from domain knowledge and dataset distributions.
    NOTE: These are qualitative indicators, NOT causal or SHAP values.
    """
    factors = []
    if features.get('deposit_type') == 'Non Refund':
        factors.append({
            'factor': 'Non-Refundable Deposit',
            'impact': 'High Risk (+35%)',
            'direction': 'risk',
            'detail': 'Non-refundable agency bookings carry high historical cancellation and default rates in this dataset.'
        })
    if features.get('lead_time', 0) > 150:
        factors.append({
            'factor': f"High Lead Time ({features['lead_time']} days)",
            'impact': 'Elevated Risk (+20%)',
            'direction': 'risk',
            'detail': 'Reservations made months in advance have high attrition due to evolving guest plans.'
        })
    elif features.get('lead_time', 0) < 14:
        factors.append({
            'factor': f"Short Lead Time ({features['lead_time']} days)",
            'impact': 'Low Risk (-15%)',
            'direction': 'safe',
            'detail': 'Last-minute bookings materialize at a significantly higher rate.'
        })
        
    if features.get('previous_cancellations', 0) > 0:
        factors.append({
            'factor': f"Prior Cancellations ({features['previous_cancellations']})",
            'impact': 'Elevated Risk (+25%)',
            'direction': 'risk',
            'detail': 'Guest profile shows a history of previous booking cancellations.'
        })
        
    if features.get('required_car_parking_spaces', 0) > 0:
        factors.append({
            'factor': 'Parking Space Requested',
            'impact': 'Strong Retention (-25%)',
            'direction': 'safe',
            'detail': 'Drive-in travelers requesting parking rarely cancel their reservations.'
        })
        
    if features.get('total_of_special_requests', 0) > 0:
        factors.append({
            'factor': f"{features['total_of_special_requests']} Special Request(s)",
            'impact': 'Low Risk (-15%)',
            'direction': 'safe',
            'detail': 'Engaged guests who communicate requests exhibit high intent to complete their stay.'
        })
        
    if features.get('is_repeated_guest') == 1:
        factors.append({
            'factor': 'Repeated Guest',
            'impact': 'Loyal Guest (-18%)',
            'direction': 'safe',
            'detail': 'Returning guests exhibit substantially higher commitment.'
        })
        
    if not factors:
        factors.append({
            'factor': 'Standard Booking Profile',
            'impact': 'Neutral',
            'direction': 'neutral',
            'detail': 'Reservation attributes align with median dataset averages.'
        })
    return factors

@app.route('/api/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({
            'success': False,
            'error': 'Model pipeline artifact is not loaded. Train the model using backend/train_model.py or ensure model_pipeline.joblib exists.'
        }), 503

    raw_data = request.get_json(silent=True)
    if raw_data is None:
        return jsonify({'success': False, 'error': 'Invalid JSON body or missing Content-Type: application/json'}), 400

    features, error_msg = validate_booking_input(raw_data)
    if error_msg:
        return jsonify({'success': False, 'error': error_msg}), 400

    try:
        input_df = pd.DataFrame([features])
        proba = float(pipeline.predict_proba(input_df)[0, 1])
        is_canceled = int(proba >= 0.5)

        # Documented risk tier thresholds
        # Threshold derivation: proba >= 0.65 represents top risk deciles; < 0.35 represents low baseline risk.
        if proba >= 0.65:
            risk_level = 'HIGH'
            badge_color = 'danger'
            recommendation = "High cancellation risk. Send an automated re-confirmation prompt or consider strategic room category overbooking."
        elif proba >= 0.35:
            risk_level = 'MEDIUM'
            badge_color = 'warning'
            recommendation = "Moderate cancellation risk. Send a pre-arrival welcome email 48 hours prior to check-in to confirm arrival time."
        else:
            risk_level = 'LOW'
            badge_color = 'success'
            recommendation = "Low cancellation risk. Reservation is highly secure and expected to proceed smoothly."

        factors = compute_heuristic_factors(features)

        return jsonify({
            'success': True,
            'stage': 'Stage 1: Binary Cancellation Prediction',
            'stage_2_note': 'Time-to-cancellation / duration prediction is not modeled by this Stage 1 classifier.',
            'prediction': is_canceled,
            'prediction_label': 'Will Cancel' if is_canceled == 1 else 'Will Not Cancel',
            'probability': round(proba, 4),
            'probability_percentage': round(proba * 100, 1),
            'risk_level': risk_level,
            'badge_color': badge_color,
            'probability_type': 'raw_ensemble_probability',
            'calibration_note': 'Raw tree ensemble probability; uncalibrated. Risk tiers derived via fixed decision thresholds.',
            'explanation_type': 'heuristic_risk_indicator',
            'explanation_note': 'Heuristic indicators derived from dataset distributions. Not causal SHAP values.',
            'contributing_factors': factors,
            'recommendation': recommendation
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f"Inference execution failed: {str(e)}"}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    if pipeline is None:
        return jsonify({
            'success': False,
            'error': 'Model pipeline artifact is not loaded. Train the model using backend/train_model.py or ensure model_pipeline.joblib exists.'
        }), 503

    payload = request.get_json(silent=True)
    if not payload or 'bookings' not in payload or not isinstance(payload['bookings'], list):
        return jsonify({'success': False, 'error': "Batch request must contain a 'bookings' array."}), 400

    bookings = payload['bookings']
    if len(bookings) == 0:
        return jsonify({'success': False, 'error': "'bookings' array cannot be empty."}), 400
    if len(bookings) > 100:
        return jsonify({'success': False, 'error': "Batch size exceeds maximum limit of 100 bookings."}), 400

    validated_list = []
    for idx, b in enumerate(bookings):
        clean, err = validate_booking_input(b)
        if err:
            return jsonify({'success': False, 'error': f"Error in booking index {idx}: {err}"}), 400
        validated_list.append(clean)

    try:
        input_df = pd.DataFrame(validated_list)
        probas = pipeline.predict_proba(input_df)[:, 1]

        results = []
        for idx, p in enumerate(probas):
            proba = float(p)
            is_canceled = int(proba >= 0.5)
            risk = 'HIGH' if proba >= 0.65 else ('MEDIUM' if proba >= 0.35 else 'LOW')
            results.append({
                'index': idx,
                'prediction': is_canceled,
                'prediction_label': 'Will Cancel' if is_canceled == 1 else 'Will Not Cancel',
                'probability': round(proba, 4),
                'probability_percentage': round(proba * 100, 1),
                'risk_level': risk
            })

        return jsonify({
            'success': True,
            'count': len(results),
            'predictions': results
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f"Batch inference failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Hotel Cancellation Prediction API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
