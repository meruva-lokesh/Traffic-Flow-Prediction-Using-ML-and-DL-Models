import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load enhanced data
print("Loading traffic data...")
df = pd.read_csv('traffic_data.csv')

# Create label encoders
le_junc = LabelEncoder()
le_weather = LabelEncoder()
le_day = LabelEncoder()
le_situ = LabelEncoder()

# Encode categorical variables
df['Junction_enc'] = le_junc.fit_transform(df['Junction'])
df['Weather_enc'] = le_weather.fit_transform(df['Weather'])
df['DayOfWeek_enc'] = le_day.fit_transform(df['DayOfWeek'])
df['Situation_enc'] = le_situ.fit_transform(df['TrafficSituation'])

# Feature Engineering - Create additional features
df['VehicleDensity'] = df['TotalVehicles'] / (df['CarCount'] + df['BusCount'] + df['BikeCount'] + df['TruckCount'] + 1)
df['HeavyVehicleRatio'] = (df['BusCount'] + df['TruckCount']) / (df['TotalVehicles'] + 1)
df['LightVehicleRatio'] = (df['CarCount'] + df['BikeCount']) / (df['TotalVehicles'] + 1)
df['CarBikeRatio'] = df['CarCount'] / (df['BikeCount'] + 1)

# Create TimeOfDay feature properly
def get_time_of_day(hour):
    if 0 <= hour < 6:
        return 0
    elif 6 <= hour < 12:
        return 1
    elif 12 <= hour < 18:
        return 2
    else:
        return 3

df['TimeOfDay'] = df['Hour'].apply(get_time_of_day)

# Interaction features
df['Weather_Hour_Interaction'] = df['Weather_enc'] * df['Hour']
df['Junction_RushHour'] = df['Junction_enc'] * df['IsRushHour']

# Select features for model
feature_columns = [
    'Junction_enc', 'CarCount', 'BusCount', 'BikeCount', 'TruckCount',
    'TotalVehicles', 'Weather_enc', 'Temperature', 'Hour', 'DayOfWeek_enc',
    'IsRushHour', 'IsWeekend', 'VehicleDensity', 'HeavyVehicleRatio',
    'LightVehicleRatio', 'CarBikeRatio', 'TimeOfDay', 'Weather_Hour_Interaction',
    'Junction_RushHour'
]

X = df[feature_columns]
y = df['Situation_enc']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print(f"\nClass distribution in training:")
print(pd.Series(y_train).value_counts())

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest with optimized parameters
print("\nTraining Random Forest Classifier...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

clf.fit(X_train_scaled, y_train)

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Make predictions
y_pred = clf.predict(X_test_scaled)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Display results
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=le_situ.classes_, zero_division=0))

print("\nConfusion Matrix:")
print(cm)
print("\nClass Labels:", le_situ.classes_)

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*60)
print(feature_importance.head(10).to_string(index=False))

# Save all models and encoders
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

joblib.dump(clf, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_junc, 'le_junc.pkl')
joblib.dump(le_weather, 'le_weather.pkl')
joblib.dump(le_day, 'le_day.pkl')
joblib.dump(le_situ, 'le_situ.pkl')
joblib.dump(cm, 'cm.pkl')
joblib.dump(acc, 'acc.pkl')
joblib.dump(prec, 'prec.pkl')
joblib.dump(rec, 'rec.pkl')
joblib.dump(f1, 'f1.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

print("✅ All models and encoders saved successfully!")
print("\nSaved files:")
print("  - rf_model.pkl (Random Forest Model)")
print("  - scaler.pkl (Feature Scaler)")
print("  - le_junc.pkl, le_weather.pkl, le_day.pkl, le_situ.pkl (Encoders)")
print("  - cm.pkl, acc.pkl, prec.pkl, rec.pkl, f1.pkl (Metrics)")
print("  - feature_columns.pkl (Feature list)")
