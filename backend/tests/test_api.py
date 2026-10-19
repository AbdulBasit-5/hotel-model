import os
import json
import pytest
from backend.app import app, load_model_pipeline
import backend.app as app_module

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Verify health check returns status, model metadata, and stage 1 labeling."""
    res = client.get('/api/health')
    assert res.status_code == 200
    data = res.get_json()
    assert data['status'] == 'healthy'
    assert 'model_loaded' in data
    assert 'stage' in data
    assert 'Stage 1' in data['stage']
    assert 'stage_2_status' in data
    assert 'Future Work' in data['stage_2_status']

def test_overview_endpoint(client):
    """Verify research overview metadata is loaded."""
    res = client.get('/api/overview')
    assert res.status_code == 200
    data = res.get_json()
    assert data['total_rows'] == 119390
    assert data['duplicate_rows'] == 31994
    assert 'protocols' in data
    assert len(data['protocols']) == 3

def test_models_endpoint(client):
    """Verify models metadata returns deployed and benchmark model list."""
    res = client.get('/api/models')
    assert res.status_code == 200
    data = res.get_json()
    assert 'deployed_model' in data
    assert 'benchmark_models_evaluated' in data
    assert 'TabNet' in data['benchmark_models_evaluated']

def test_benchmarks_and_metrics_endpoints(client):
    """Verify benchmarks and metrics alias return authentic benchmark rows with filters."""
    res = client.get('/api/benchmarks')
    assert res.status_code == 200
    data = res.get_json()
    assert data['count'] >= 30

    # Test alias route /api/metrics
    res_alias = client.get('/api/metrics')
    assert res_alias.status_code == 200
    assert res_alias.get_json()['count'] == data['count']

    # Test hotel filtering
    res_h1 = client.get('/api/benchmarks?hotel=H1')
    assert res_h1.status_code == 200
    for row in res_h1.get_json()['benchmarks']:
        assert 'H1' in row['hotel']

    # Test protocol filtering
    res_p3 = client.get('/api/benchmarks?protocol=P3')
    assert res_p3.status_code == 200
    for row in res_p3.get_json()['benchmarks']:
        assert 'P3' in row['protocol']

def test_tabnet_comparison_endpoint(client):
    """Verify Gate 1 comparison table exists with published and local numbers."""
    res = client.get('/api/tabnet-comparison')
    assert res.status_code == 200
    data = res.get_json()
    assert 'comparisons' in data
    assert len(data['comparisons']) == 2
    assert data['comparisons'][0]['published_auc_roc'] == 0.862

def test_ablation_endpoint(client):
    """Verify ablation study results are returned."""
    res = client.get('/api/ablation')
    assert res.status_code == 200
    data = res.get_json()
    assert 'ablation' in data
    assert len(data['ablation']) == 4

def test_feature_defaults_endpoint(client):
    """Verify defaults and options are returned."""
    res = client.get('/api/feature-defaults')
    assert res.status_code == 200
    data = res.get_json()
    assert 'defaults' in data
    assert 'options' in data
    assert 'City Hotel' in data['options']['hotel']

def test_predict_valid_booking(client):
    """Verify single prediction with valid input returns genuine probability and risk tier."""
    payload = {
        'hotel': 'City Hotel',
        'lead_time': 120,
        'deposit_type': 'No Deposit',
        'adr': 105.0,
        'adults': 2,
        'market_segment': 'Online TA',
        'total_of_special_requests': 1,
        'required_car_parking_spaces': 0
    }
    res = client.post('/api/predict', json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert 0.0 <= data['probability'] <= 1.0
    assert data['prediction'] in [0, 1]
    assert data['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']
    assert data['explanation_type'] == 'heuristic_risk_indicator'
    assert 'heuristic' in data['explanation_note'].lower()
    assert 'Stage 1' in data['stage']

def test_predict_invalid_numeric_values(client):
    """Verify rejection of invalid numeric values (negative lead_time, negative adr)."""
    # Negative lead time
    res = client.post('/api/predict', json={'lead_time': -15})
    assert res.status_code == 400
    assert 'out of bounds' in res.get_json()['error']

    # Negative ADR
    res2 = client.post('/api/predict', json={'adr': -50.0})
    assert res2.status_code == 400
    assert 'out of bounds' in res2.get_json()['error']

    # Non-numeric value for numeric field
    res3 = client.post('/api/predict', json={'adults': 'three_people'})
    assert res3.status_code == 400
    assert 'must be a valid int' in res3.get_json()['error']

def test_predict_invalid_categories(client):
    """Verify rejection of unsupported categories."""
    res = client.post('/api/predict', json={'hotel': 'Space Hotel Mars'})
    assert res.status_code == 400
    assert 'Invalid value' in res.get_json()['error']

    res2 = client.post('/api/predict', json={'deposit_type': 'Crypto Deposit'})
    assert res2.status_code == 400
    assert 'deposit_type' in res2.get_json()['error']

def test_predict_empty_or_malformed_json(client):
    """Verify rejection of missing or malformed JSON body."""
    res = client.post('/api/predict', data="not json", content_type='text/plain')
    assert res.status_code == 400

def test_predict_batch_valid(client):
    """Verify batch prediction returns predictions for each item."""
    batch_payload = {
        'bookings': [
            {'hotel': 'City Hotel', 'lead_time': 200, 'deposit_type': 'Non Refund'},
            {'hotel': 'Resort Hotel', 'lead_time': 10, 'deposit_type': 'No Deposit', 'required_car_parking_spaces': 1}
        ]
    }
    res = client.post('/api/predict/batch', json=batch_payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert data['count'] == 2
    assert len(data['predictions']) == 2
    assert data['predictions'][0]['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']

def test_predict_batch_invalid(client):
    """Verify batch rejection on invalid format or invalid item."""
    # Missing bookings array
    res = client.post('/api/predict/batch', json={'not_bookings': []})
    assert res.status_code == 400

    # Empty bookings array
    res2 = client.post('/api/predict/batch', json={'bookings': []})
    assert res2.status_code == 400

    # Invalid item inside batch
    res3 = client.post('/api/predict/batch', json={'bookings': [{'lead_time': -10}]})
    assert res3.status_code == 400
    assert 'booking index 0' in res3.get_json()['error']

def test_model_unavailable_returns_503(client, monkeypatch):
    """Verify that if the model pipeline is None, /api/predict returns 503 instead of fake predictions."""
    monkeypatch.setattr(app_module, 'pipeline', None)

    res = client.post('/api/predict', json={'lead_time': 50})
    assert res.status_code == 503
    assert res.get_json()['success'] is False
    assert 'not loaded' in res.get_json()['error']

    res_batch = client.post('/api/predict/batch', json={'bookings': [{'lead_time': 50}]})
    assert res_batch.status_code == 503
    assert res_batch.get_json()['success'] is False
