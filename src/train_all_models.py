import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Import all models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print("="*70)
print("MULTI-MODEL TRAFFIC FLOW PREDICTION TRAINING")
print("="*70)

# Load enhanced data
print("\n📂 Loading traffic data...")
df = pd.read_csv('traffic_data.csv')
print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")

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

# Feature Engineering
print("\n🔧 Engineering features...")
df['VehicleDensity'] = df['TotalVehicles'] / (df['CarCount'] + df['BusCount'] + df['BikeCount'] + df['TruckCount'] + 1)
df['HeavyVehicleRatio'] = (df['BusCount'] + df['TruckCount']) / (df['TotalVehicles'] + 1)
df['LightVehicleRatio'] = (df['CarCount'] + df['BikeCount']) / (df['TotalVehicles'] + 1)
df['CarBikeRatio'] = df['CarCount'] / (df['BikeCount'] + 1)

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
df['Weather_Hour_Interaction'] = df['Weather_enc'] * df['Hour']
df['Junction_RushHour'] = df['Junction_enc'] * df['IsRushHour']

# Select features
feature_columns = [
    'Junction_enc', 'CarCount', 'BusCount', 'BikeCount', 'TruckCount',
    'TotalVehicles', 'Weather_enc', 'Temperature', 'Hour', 'DayOfWeek_enc',
    'IsRushHour', 'IsWeekend', 'VehicleDensity', 'HeavyVehicleRatio',
    'LightVehicleRatio', 'CarBikeRatio', 'TimeOfDay', 'Weather_Hour_Interaction',
    'Junction_RushHour'
]

X = df[feature_columns]
y = df['Situation_enc']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training set: {len(X_train)} samples")
print(f"✅ Testing set: {len(X_test)} samples")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define all models
print("\n🤖 Initializing 5 Machine Learning Models...")
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(
        C=1.0,
        kernel='rbf',
        random_state=42,
        probability=True,
        class_weight='balanced'
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=15,
        min_samples_leaf=8,
        random_state=42,
        class_weight='balanced'
    )
}

# Storage for results
results = {}
model_files = {}

print("\n" + "="*70)
print("TRAINING ALL MODELS")
print("="*70)

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🔄 Training: {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    # Save model
    model_filename = f"model_{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, model_filename)
    model_files[name] = model_filename
    
    print(f"   ✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"   ✅ F1-Score: {f1:.4f}")
    print(f"   ✅ CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   💾 Saved as: {model_filename}")

# Display comparison
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()],
    'CV Mean': [results[m]['cv_mean'] for m in results.keys()],
    'CV Std': [results[m]['cv_std'] for m in results.keys()]
}).sort_values('Accuracy', ascending=False)

print("\n" + comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.iloc[0]['Model']
best_accuracy = comparison_df.iloc[0]['Accuracy']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Save comparison results
joblib.dump(comparison_df, 'model_comparison.pkl')
joblib.dump(results, 'all_model_results.pkl')

# Save other required files
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_junc, 'le_junc.pkl')
joblib.dump(le_weather, 'le_weather.pkl')
joblib.dump(le_day, 'le_day.pkl')
joblib.dump(le_situ, 'le_situ.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

# Save best model's confusion matrix and metrics
best_model_results = results[best_model_name]
joblib.dump(best_model_results['confusion_matrix'], 'cm.pkl')
joblib.dump(best_model_results['accuracy'], 'acc.pkl')
joblib.dump(best_model_results['precision'], 'prec.pkl')
joblib.dump(best_model_results['recall'], 'rec.pkl')
joblib.dump(best_model_results['f1_score'], 'f1.pkl')

print("\n" + "="*70)
print("SAVING ALL MODELS AND ENCODERS")
print("="*70)
print("\n✅ Models saved:")
for name, filename in model_files.items():
    print(f"   - {filename}")

print("\n✅ Encoders and scalers saved:")
print("   - scaler.pkl")
print("   - le_junc.pkl, le_weather.pkl, le_day.pkl, le_situ.pkl")
print("   - feature_columns.pkl")

print("\n✅ Results saved:")
print("   - model_comparison.pkl (comparison DataFrame)")
print("   - all_model_results.pkl (detailed results)")
print("   - cm.pkl, acc.pkl, prec.pkl, rec.pkl, f1.pkl (best model metrics)")

# Create visualization
print("\n📊 Creating comparison visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Accuracy comparison
ax1 = axes[0, 0]
models_list = comparison_df['Model'].tolist()
accuracies = comparison_df['Accuracy'].tolist()
colors = ['green' if m == best_model_name else 'steelblue' for m in models_list]
bars = ax1.barh(models_list, accuracies, color=colors)
ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim([0, 1])
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(acc + 0.01, i, f'{acc:.4f}', va='center', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 2. All metrics comparison
ax2 = axes[0, 1]
metrics_data = comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].set_index('Model')
metrics_data.plot(kind='bar', ax=ax2, width=0.8)
ax2.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_xlabel('')
ax2.legend(loc='lower right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 1])
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 3. Cross-validation scores
ax3 = axes[1, 0]
cv_means = comparison_df['CV Mean'].tolist()
cv_stds = comparison_df['CV Std'].tolist()
ax3.barh(models_list, cv_means, xerr=cv_stds, color='coral', alpha=0.7)
ax3.set_xlabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Cross-Validation Performance (5-fold)', fontsize=14, fontweight='bold')
ax3.set_xlim([0, 1])
for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    ax3.text(mean + 0.01, i, f'{mean:.4f}±{std:.4f}', va='center', fontsize=9)
ax3.grid(axis='x', alpha=0.3)

# 4. Best model confusion matrix
ax4 = axes[1, 1]
cm_best = best_model_results['confusion_matrix']
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=le_situ.classes_, yticklabels=le_situ.classes_)
ax4.set_title(f'{best_model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
ax4.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax4.set_ylabel('Actual', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison_charts.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved visualization: model_comparison_charts.png")

print("\n" + "="*70)
print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("="*70)
print(f"\n🎯 You can now run the multi-model app with:")
print("   streamlit run app_multi_model.py")
