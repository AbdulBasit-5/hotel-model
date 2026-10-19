import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import joblib
import os
import time

print("Loading hotel_bookings.csv...")
df = pd.read_csv('hotel_bookings.csv')
print(f"Loaded {len(df)} rows.")

TARGET = 'is_canceled'
ALWAYS_DROP = ['reservation_status', 'reservation_status_date', 'is_canceled', 'arrival_date_year', 'arrival_date']

CAT_COLS = [
    'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
    'distribution_channel', 'reserved_room_type', 'assigned_room_type',
    'deposit_type', 'customer_type', 'agent', 'company'
]

# Ensure agent and company are strings
for c in ['agent', 'company']:
    if c in df.columns:
        df[c] = df[c].fillna('-1').astype(str)

feature_cols = [c for c in df.columns if c not in ALWAYS_DROP]
cat_features = [c for c in CAT_COLS if c in feature_cols]
num_features = [c for c in feature_cols if c not in cat_features]

print(f"Features: {len(feature_cols)} total ({len(cat_features)} categorical, {len(num_features)} numerical)")

X = df[feature_cols]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, cat_features),
        ('num', num_transformer, num_features)
    ]
)

model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.08,
    max_leaf_nodes=31,
    random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

print("Training HistGradientBoostingClassifier...")
t0 = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Trained in {train_time:.2f} seconds.")

y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Test AUC-ROC: {auc:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1: {f1:.4f}")

# Save artifacts
os.makedirs('backend', exist_ok=True)
metadata = {
    'feature_cols': feature_cols,
    'cat_features': cat_features,
    'num_features': num_features,
    'metrics': {
        'auc_roc': round(auc, 4),
        'accuracy': round(acc, 4),
        'f1': round(f1, 4),
        'train_time': round(train_time, 2)
    }
}

joblib.dump(pipeline, 'backend/model_pipeline.joblib')
with open('backend/model_metadata.json', 'w', encoding='utf-8') as f:
    import json
    json.dump(metadata, f, indent=2)

print("Saved backend/model_pipeline.joblib and backend/model_metadata.json successfully!")
