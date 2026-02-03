"""
Enhanced CNN Model Training for Higher Accuracy
Run: python src/train_enhanced_cnn.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.regularizers import l2

print("="*70)
print("ENHANCED 1D CNN TRAINING FOR MAXIMUM ACCURACY")
print("="*70)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
data_path = Path("data/traffic_data.csv")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("="*70)

df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df)} samples")

# Feature Engineering
print("\n📊 Feature Engineering...")

# Time of day
def get_time_of_day(hour):
    if 0 <= hour < 6: return 0
    elif 6 <= hour < 12: return 1
    elif 12 <= hour < 18: return 2
    else: return 3

df['TimeOfDay'] = df['Hour'].apply(get_time_of_day)

# Vehicle features
df['VehicleDensity'] = df['TotalVehicles'] / (df['CarCount'] + df['BusCount'] + df['BikeCount'] + df['TruckCount'] + 1)
df['HeavyVehicleRatio'] = (df['BusCount'] + df['TruckCount']) / (df['TotalVehicles'] + 1)
df['LightVehicleRatio'] = (df['CarCount'] + df['BikeCount']) / (df['TotalVehicles'] + 1)
df['CarBikeRatio'] = df['CarCount'] / (df['BikeCount'] + 1)

# Encode categorical
le_junc = LabelEncoder()
le_weather = LabelEncoder()
le_day = LabelEncoder()
le_situ = LabelEncoder()

df['Junction_enc'] = le_junc.fit_transform(df['Junction'])
df['Weather_enc'] = le_weather.fit_transform(df['Weather'])
df['DayOfWeek_enc'] = le_day.fit_transform(df['DayOfWeek'])
df['Situation_enc'] = le_situ.fit_transform(df['TrafficSituation'])

# Interaction features
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

X = df[feature_columns].values
y = df['Situation_enc'].values
num_classes = len(np.unique(y))

print(f"✓ Features: {X.shape[1]}")
print(f"✓ Classes: {num_classes}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to categorical
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Reshape for CNN
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"✓ Training: {X_train_cnn.shape[0]} samples")
print(f"✓ Testing: {X_test_cnn.shape[0]} samples")
print(f"✓ Input shape: {X_train_cnn.shape[1:]}")

# ============================================================================
# STEP 2: BUILD ENHANCED CNN MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 2: BUILDING ENHANCED 1D CNN MODEL")
print("="*70)

input_dim = X_train_cnn.shape[1]

def build_enhanced_cnn():
    """
    Enhanced 1D CNN with:
    - More convolutional layers
    - Residual-like connections
    - Better regularization
    - Optimized architecture
    """
    inputs = layers.Input(shape=(input_dim, 1))
    
    # Block 1
    x = layers.Conv1D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 4 (Additional deep block)
    x = layers.Conv1D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    
    # Dense layers
    x = layers.Dense(512, kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = build_enhanced_cnn()
print(f"✓ Enhanced CNN Model built")

# Print architecture
model.summary()

# ============================================================================
# STEP 3: COMPILE MODEL WITH OPTIMAL SETTINGS
# ============================================================================
print("\n" + "="*70)
print("STEP 3: COMPILING MODEL")
print("="*70)

# Use Adam optimizer with optimal learning rate
optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled with Adam optimizer")

# ============================================================================
# STEP 4: SETUP CALLBACKS
# ============================================================================
print("\n" + "="*70)
print("STEP 4: SETTING UP CALLBACKS")
print("="*70)

callbacks = [
    # Stop if no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    ),
    # Save best model
    ModelCheckpoint(
        str(models_dir / 'enhanced_cnn_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("✓ Callbacks configured:")
print("   - Early stopping (patience=20)")
print("   - Learning rate reduction")
print("   - Model checkpoint")

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 5: TRAINING ENHANCED CNN MODEL")
print("="*70)

print("\n🚀 Training started...")
print("   This may take 10-15 minutes")

history = model.fit(
    X_train_cnn, y_train_cat,
    validation_split=0.2,
    epochs=200,  # More epochs with early stopping
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training completed!")

# ============================================================================
# STEP 6: EVALUATE MODEL
# ============================================================================
print("\n" + "="*70)
print("STEP 6: MODEL EVALUATION")
print("="*70)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_cat, verbose=0)

# Predictions
y_pred_probs = model.predict(X_test_cnn, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted', zero_division=0
)

print("\n📊 RESULTS:")
print("-" * 70)
print(f"Test Accuracy:     {test_accuracy*100:.2f}%")
print(f"Precision:         {precision*100:.2f}%")
print(f"Recall:            {recall*100:.2f}%")
print(f"F1-Score:          {f1*100:.2f}%")
print("-" * 70)

# Best validation accuracy
best_val_acc = max(history.history['val_accuracy'])
print(f"\nBest Val Accuracy: {best_val_acc*100:.2f}%")

# Training history
final_train_acc = history.history['accuracy'][-1]
print(f"Final Train Accuracy: {final_train_acc*100:.2f}%")

# ============================================================================
# STEP 7: SAVE MODEL AND RESULTS
# ============================================================================
print("\n" + "="*70)
print("STEP 7: SAVING MODEL AND RESULTS")
print("="*70)

# Save final model
model.save(models_dir / 'enhanced_cnn_final.h5')
print("✓ Saved: models/enhanced_cnn_final.h5")
print("✓ Saved: models/enhanced_cnn_best.h5 (best during training)")

# Save results
results = {
    'test_accuracy': float(test_accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'best_val_accuracy': float(best_val_acc),
    'final_train_accuracy': float(final_train_acc)
}

import json
with open(models_dir / 'enhanced_cnn_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Saved: models/enhanced_cnn_results.json")

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("STEP 8: CREATING VISUALIZATIONS")
print("="*70)

# Training history plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Enhanced CNN - Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Enhanced CNN - Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(models_dir / 'enhanced_cnn_training.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/enhanced_cnn_training.png")
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High', 'Severe'],
            yticklabels=['Low', 'Medium', 'High', 'Severe'])
plt.title('Enhanced CNN - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(models_dir / 'enhanced_cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/enhanced_cnn_confusion_matrix.png")
plt.close()

# ============================================================================
# COMPARISON WITH ORIGINAL CNN
# ============================================================================
print("\n" + "="*70)
print("COMPARISON WITH ORIGINAL CNN")
print("="*70)

try:
    with open(models_dir / 'deep_learning_results.json', 'r') as f:
        dl_results = json.load(f)
    
    if '1D CNN' in dl_results:
        original_acc = dl_results['1D CNN']['test_accuracy'] * 100
        improvement = test_accuracy * 100 - original_acc
        
        print(f"\nOriginal CNN Accuracy:  {original_acc:.2f}%")
        print(f"Enhanced CNN Accuracy:  {test_accuracy*100:.2f}%")
        print(f"Improvement:            {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"\n✅ Enhanced CNN is {improvement:.2f}% better!")
        else:
            print(f"\n⚠️ Original CNN was better by {-improvement:.2f}%")
    
except:
    print("\n⚠️ Could not load original CNN results for comparison")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)

print(f"\n✅ Enhanced CNN Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✅ F1-Score: {f1*100:.2f}%")

print("\n📁 Files saved:")
print("   • models/enhanced_cnn_final.h5")
print("   • models/enhanced_cnn_best.h5")
print("   • models/enhanced_cnn_results.json")
print("   • models/enhanced_cnn_training.png")
print("   • models/enhanced_cnn_confusion_matrix.png")

print("\n📊 Next steps:")
print("   1. Run: python check_accuracy.py (to compare all models)")
print("   2. Use the best model in your Streamlit app")
print("   3. If accuracy is still low, try:")
print("      - Generate more training data")
print("      - Tune hyperparameters further")
print("      - Try ensemble methods")

print("\n" + "="*70)
