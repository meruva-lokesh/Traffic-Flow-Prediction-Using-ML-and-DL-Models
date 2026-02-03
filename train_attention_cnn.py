"""
Train the Novel Attention-Enhanced CNN
This will use your unique contribution for publication
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import sys
from pathlib import Path

# Import your novel attention CNN
sys.path.append(str(Path(__file__).parent))
from src.attention_cnn import create_attention_cnn, visualize_attention_weights

print("="*80)
print("🎯 ATTENTION-ENHANCED CNN TRAINING")
print("Novel Architecture with Self-Attention Mechanism")
print("="*80)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('data/traffic_data.csv')
print(f"✓ Loaded {len(df)} samples")

# Encode and engineer features (same as train_stable_publication.py)
le_junction = LabelEncoder()
le_weather = LabelEncoder()
le_dayofweek = LabelEncoder()
le_trafficsituation = LabelEncoder()
le_timeofday = LabelEncoder()

df['Junction_enc'] = le_junction.fit_transform(df['Junction'])
df['Weather_enc'] = le_weather.fit_transform(df['Weather'])
df['DayOfWeek_enc'] = le_dayofweek.fit_transform(df['DayOfWeek'])
df['TrafficSituation_enc'] = le_trafficsituation.fit_transform(df['TrafficSituation'])

# Feature Engineering
df['VehicleDensity'] = df['TotalVehicles'] / (df['CarCount'] + df['BikeCount'] + 1)
df['HeavyVehicleRatio'] = (df['BusCount'] + df['TruckCount']) / (df['TotalVehicles'] + 1)

def get_time_of_day(hour):
    if 0 <= hour < 6: return 0
    elif 6 <= hour < 12: return 1
    elif 12 <= hour < 18: return 2
    else: return 3

df['TimeOfDay'] = df['Hour'].apply(get_time_of_day)
df['TimeOfDay_enc'] = le_timeofday.fit_transform(df['TimeOfDay'].astype(str))
df['IsRushHour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9) | (df['Hour'] >= 17) & (df['Hour'] <= 19)).astype(int)
df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)
df['Weather_Hour_Interaction'] = df['Weather_enc'] * df['Hour']
df['Junction_RushHour'] = df['Junction_enc'] * df['IsRushHour']

# Select features (17 features total)
feature_cols = ['Junction_enc', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 
                'TotalVehicles', 'Weather_enc', 'Temperature', 'Hour', 'DayOfWeek_enc',
                'VehicleDensity', 'HeavyVehicleRatio', 'TimeOfDay_enc', 'IsRushHour',
                'IsWeekend', 'Weather_Hour_Interaction', 'Junction_RushHour']

X = df[feature_cols].values
y_raw = df['TrafficSituation_enc'].values

# Encode labels
y = y_raw
n_classes = len(np.unique(y))

print(f"✓ Features: {X.shape[1]}")
print(f"✓ Classes: {n_classes}")

# 5-Fold Cross-Validation
print("\n" + "="*80)
print("RUNNING 5-FOLD CROSS-VALIDATION FOR ATTENTION CNN")
print("="*80)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
attention_results = {
    'accuracies': [],
    'precisions': [],
    'recalls': [],
    'f1s': []
}

for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
    print(f"\n{'='*80}")
    print(f"FOLD {fold}/5")
    print("="*80)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for CNN (add channel dimension)
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, n_classes)
    
    # DATA AUGMENTATION - Same as optimized CNN
    print("  Applying data augmentation...")
    augmented_X = [X_train_cnn]
    augmented_y = [y_train_cat]
    
    # Add Gaussian noise variations
    for noise_level in [0.02, 0.04]:
        X_noisy = X_train_cnn + np.random.normal(0, noise_level, X_train_cnn.shape)
        augmented_X.append(X_noisy)
        augmented_y.append(y_train_cat)
    
    X_train_aug = np.vstack(augmented_X)
    y_train_aug = np.vstack(augmented_y)
    print(f"  ✓ Augmented training data: {X_train_aug.shape[0]} samples (3x original)")
    
    # Create NOVEL Attention CNN
    print("  Building Attention-Enhanced CNN...")
    attention_model = create_attention_cnn(
        input_shape=(X_train_cnn.shape[1], 1),
        n_classes=n_classes
    )
    
    # Compile with optimized settings
    attention_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=0)
    
    # Train
    print("  Training Attention CNN...")
    attention_model.fit(
        X_train_aug, y_train_aug,
        validation_split=0.2,
        epochs=200,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Evaluate
    y_pred_proba = attention_model.predict(X_test_cnn, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    attention_results['accuracies'].append(acc)
    attention_results['precisions'].append(prec)
    attention_results['recalls'].append(rec)
    attention_results['f1s'].append(f1)
    
    print(f"  ✓ Attention CNN: {acc*100:.2f}%")
    
    # Visualize attention for last fold
    if fold == 5:
        print("\n  Generating attention visualization...")
        Path('models').mkdir(exist_ok=True)
        visualize_attention_weights(
            attention_model,
            X_test_cnn,
            save_path='models/attention_weights_visualization.png'
        )

# Calculate final statistics
print("\n" + "="*80)
print("FINAL RESULTS: ATTENTION-ENHANCED CNN")
print("="*80)

mean_acc = np.mean(attention_results['accuracies']) * 100
std_acc = np.std(attention_results['accuracies']) * 100
mean_f1 = np.mean(attention_results['f1s']) * 100
std_f1 = np.std(attention_results['f1s']) * 100

print(f"\nAttention CNN:")
print(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
print(f"  F1-Score: {mean_f1:.2f}% ± {std_f1:.2f}%")
print(f"  Precision: {np.mean(attention_results['precisions'])*100:.2f}%")
print(f"  Recall: {np.mean(attention_results['recalls'])*100:.2f}%")

# Save results
results_dir = Path('publication_results')
results_dir.mkdir(exist_ok=True)

attention_df = pd.DataFrame({
    'Model': ['Attention CNN'],
    'Accuracy Mean (%)': [mean_acc],
    'Accuracy Std (%)': [std_acc],
    'F1-Score (%)': [mean_f1],
    'F1 Std (%)': [std_f1]
})

attention_df.to_csv(results_dir / 'attention_cnn_results.csv', index=False)

print("\n" + "="*80)
print("✅ ATTENTION CNN TRAINING COMPLETE!")
print("="*80)
print(f"\n📊 Results saved to: {results_dir}/attention_cnn_results.csv")
print(f"📈 Attention visualization: models/attention_weights_visualization.png")
print("\n💡 This is your NOVEL contribution for publication!")
print("   Use these results in your conference paper as the innovative component.")
