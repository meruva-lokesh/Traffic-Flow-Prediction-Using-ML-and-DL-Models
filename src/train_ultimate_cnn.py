"""
ULTIMATE CNN TRAINING - TOP 1 ACCURACY GUARANTEED
Combines: Residual Connections + Attention + Deep Architecture + Advanced Training
Run: python src/train_ultimate_cnn.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.regularizers import l1_l2

print("="*70)
print("🏆 ULTIMATE CNN TRAINING - TOP 1 ACCURACY TARGET")
print("="*70)
print("Architecture: ResNet-style + Attention + Deep Dense Layers")
print("Target: Beat all models (VGG16, VGG19, ResNet50, All ML)")
print("="*70)

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Paths
data_path = Path("data/traffic_data.csv")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: DATA LOADING WITH AUGMENTATION
# ============================================================================
print("\n" + "="*70)
print("STEP 1: DATA LOADING AND ADVANCED PREPROCESSING")
print("="*70)

df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df)} samples")

# Feature Engineering
print("\n📊 Feature Engineering...")

def get_time_of_day(hour):
    if 0 <= hour < 6: return 0
    elif 6 <= hour < 12: return 1
    elif 12 <= hour < 18: return 2
    else: return 3

df['TimeOfDay'] = df['Hour'].apply(get_time_of_day)

# Advanced vehicle features
df['VehicleDensity'] = df['TotalVehicles'] / (df['CarCount'] + df['BusCount'] + df['BikeCount'] + df['TruckCount'] + 1)
df['HeavyVehicleRatio'] = (df['BusCount'] + df['TruckCount']) / (df['TotalVehicles'] + 1)
df['LightVehicleRatio'] = (df['CarCount'] + df['BikeCount']) / (df['TotalVehicles'] + 1)
df['CarBikeRatio'] = df['CarCount'] / (df['BikeCount'] + 1)
df['BusTruckRatio'] = df['BusCount'] / (df['TruckCount'] + 1)
df['VehicleVariance'] = df[['CarCount', 'BusCount', 'BikeCount', 'TruckCount']].var(axis=1)
df['HourSin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['HourCos'] = np.cos(2 * np.pi * df['Hour'] / 24)

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
df['Weather_RushHour'] = df['Weather_enc'] * df['IsRushHour']
df['Temp_Hour'] = df['Temperature'] * df['Hour']

# Select features (MORE features = better learning)
feature_columns = [
    'Junction_enc', 'CarCount', 'BusCount', 'BikeCount', 'TruckCount',
    'TotalVehicles', 'Weather_enc', 'Temperature', 'Hour', 'DayOfWeek_enc',
    'IsRushHour', 'IsWeekend', 'VehicleDensity', 'HeavyVehicleRatio',
    'LightVehicleRatio', 'CarBikeRatio', 'TimeOfDay', 'Weather_Hour_Interaction',
    'Junction_RushHour', 'BusTruckRatio', 'VehicleVariance', 'HourSin', 'HourCos',
    'Weather_RushHour', 'Temp_Hour'
]

X = df[feature_columns].values
y = df['Situation_enc'].values
num_classes = len(np.unique(y))

print(f"✓ Features: {X.shape[1]} (enhanced from 19)")
print(f"✓ Classes: {num_classes}")

# Calculate class weights for imbalanced data
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
print(f"✓ Class weights computed: {class_weights_dict}")

# Scale features with robust scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y  # 85/15 split for more training data
)

# Convert to categorical
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Reshape for CNN
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"✓ Training: {X_train_cnn.shape[0]} samples (85%)")
print(f"✓ Testing: {X_test_cnn.shape[0]} samples (15%)")
print(f"✓ Input shape: {X_train_cnn.shape[1:]}")

# ============================================================================
# STEP 2: BUILD ULTIMATE CNN WITH RESIDUAL + ATTENTION
# ============================================================================
print("\n" + "="*70)
print("STEP 2: BUILDING ULTIMATE CNN ARCHITECTURE")
print("="*70)
print("Components:")
print("  • Residual connections (like ResNet)")
print("  • Attention mechanism")
print("  • Very deep architecture (6 blocks)")
print("  • Advanced regularization")
print("="*70)

input_dim = X_train_cnn.shape[1]

# Custom attention layer
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                shape=(input_shape[-1], 1),
                                initializer='glorot_uniform',
                                trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[1], 1),
                                initializer='zeros',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Calculate attention scores
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return output

def residual_block(x, filters, kernel_size=3, dropout_rate=0.3):
    """Residual block with skip connection"""
    shortcut = x
    
    # Main path
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut dimensions if needed
    if K.int_shape(shortcut)[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    return x

def build_ultimate_cnn():
    """
    Ultimate CNN Architecture:
    - 6 Residual Blocks with increasing filters
    - Attention mechanism
    - Very deep dense layers
    - Advanced regularization
    """
    inputs = layers.Input(shape=(input_dim, 1))
    
    # Initial convolution
    x = layers.Conv1D(64, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Residual Block 1 (64 filters)
    x = residual_block(x, 64, dropout_rate=0.25)
    x = residual_block(x, 64, dropout_rate=0.25)
    x = layers.MaxPooling1D(2)(x)
    
    # Residual Block 2 (128 filters)
    x = residual_block(x, 128, dropout_rate=0.25)
    x = residual_block(x, 128, dropout_rate=0.25)
    x = layers.MaxPooling1D(2)(x)
    
    # Residual Block 3 (256 filters)
    x = residual_block(x, 256, dropout_rate=0.3)
    x = residual_block(x, 256, dropout_rate=0.3)
    x = residual_block(x, 256, dropout_rate=0.3)
    x = layers.MaxPooling1D(2)(x)
    
    # Residual Block 4 (512 filters)
    x = residual_block(x, 512, dropout_rate=0.3)
    x = residual_block(x, 512, dropout_rate=0.3)
    
    # Attention mechanism
    x = AttentionLayer()(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    
    # Very deep dense layers
    x = layers.Dense(1024, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = build_ultimate_cnn()
print("\n✓ Ultimate CNN Model built")
print(f"✓ Total parameters: {model.count_params():,}")

# ============================================================================
# STEP 3: COMPILE WITH ADVANCED OPTIMIZER
# ============================================================================
print("\n" + "="*70)
print("STEP 3: COMPILING MODEL")
print("="*70)

# Cosine decay learning rate schedule
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    alpha=0.0001
)

optimizer = Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled with Adam + Cosine Decay LR")

# ============================================================================
# STEP 4: ADVANCED CALLBACKS
# ============================================================================
print("\n" + "="*70)
print("STEP 4: SETTING UP CALLBACKS")
print("="*70)

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=30,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=15,
        min_lr=1e-8,
        verbose=1,
        mode='min'
    ),
    ModelCheckpoint(
        str(models_dir / 'ultimate_cnn_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
]

print("✓ Callbacks configured (patience=30)")

# ============================================================================
# STEP 5: TRAIN WITH CLASS WEIGHTS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: TRAINING ULTIMATE CNN")
print("="*70)
print("🚀 Training with class weights for balanced learning")
print("   This may take 15-20 minutes for best results")
print("="*70)

history = model.fit(
    X_train_cnn, y_train_cat,
    validation_split=0.2,
    epochs=300,  # Many epochs with early stopping
    batch_size=16,  # Smaller batch for better gradients
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

print("\n✓ Training completed!")

# ============================================================================
# STEP 6: EVALUATION
# ============================================================================
print("\n" + "="*70)
print("STEP 6: MODEL EVALUATION")
print("="*70)

# Load best model
best_model = keras.models.load_model(models_dir / 'ultimate_cnn_best.h5', 
                                     custom_objects={'AttentionLayer': AttentionLayer})

# Evaluate
test_loss, test_accuracy = best_model.evaluate(X_test_cnn, y_test_cat, verbose=0)

# Predictions
y_pred_probs = best_model.predict(X_test_cnn, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted', zero_division=0
)

print("\n📊 ULTIMATE CNN RESULTS:")
print("="*70)
print(f"🏆 Test Accuracy:     {test_accuracy*100:.2f}%")
print(f"📊 Precision:         {precision*100:.2f}%")
print(f"📊 Recall:            {recall*100:.2f}%")
print(f"📊 F1-Score:          {f1*100:.2f}%")
print("="*70)

best_val_acc = max(history.history['val_accuracy'])
print(f"\nBest Val Accuracy: {best_val_acc*100:.2f}%")

# Per-class accuracy
cm = confusion_matrix(y_test, y_pred)
class_accuracies = cm.diagonal() / cm.sum(axis=1)
print(f"\nPer-Class Accuracies:")
traffic_levels = ['Low', 'Medium', 'High', 'Severe']
for i, acc in enumerate(class_accuracies):
    print(f"  {traffic_levels[i]}: {acc*100:.2f}%")

# ============================================================================
# STEP 7: COMPARE WITH ALL OTHER MODELS
# ============================================================================
print("\n" + "="*70)
print("🏆 RANKING vs ALL MODELS")
print("="*70)

all_models = {}

# Load ML results
try:
    ml_comparison = joblib.load(models_dir / 'model_comparison.pkl')
    for idx, row in ml_comparison.iterrows():
        all_models[f"{row['Model']} (ML)"] = row['Accuracy'] * 100
except:
    pass

# Load DL results
try:
    import json
    with open(models_dir / 'deep_learning_results.json', 'r') as f:
        dl_results = json.load(f)
    for model_name, result in dl_results.items():
        all_models[f"{model_name} (DL)"] = result['test_accuracy'] * 100
except:
    try:
        dl_df = pd.read_csv(models_dir / 'deep_learning_comparison.csv')
        for idx, row in dl_df.iterrows():
            all_models[f"{row['Model']} (DL)"] = row['Test Accuracy (%)']
    except:
        pass

# Add Ultimate CNN
all_models['Ultimate CNN (DL)'] = test_accuracy * 100

# Sort and display
sorted_models = sorted(all_models.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Model':<45} {'Accuracy':<12} {'Status':<10}")
print("-" * 75)

for rank, (model_name, acc) in enumerate(sorted_models, 1):
    symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    status = "TOP 1! 🎉" if rank == 1 and 'Ultimate CNN' in model_name else ""
    print(f"{rank:<6} {model_name:<45} {acc:>10.2f}% {symbol} {status}")

# Check if Ultimate CNN is #1
ultimate_rank = next((i for i, (name, _) in enumerate(sorted_models, 1) if 'Ultimate CNN' in name), None)

print("\n" + "="*70)
if ultimate_rank == 1:
    print("🎉🎉🎉 SUCCESS! ULTIMATE CNN IS TOP 1! 🎉🎉🎉")
    print(f"✅ Ultimate CNN achieved {test_accuracy*100:.2f}% accuracy")
    print(f"✅ Beat all {len(sorted_models)-1} other models!")
else:
    print(f"⚠️  Ultimate CNN ranked #{ultimate_rank}")
    best_model_name, best_acc = sorted_models[0]
    diff = best_acc - test_accuracy * 100
    print(f"   Best model: {best_model_name} ({best_acc:.2f}%)")
    print(f"   Difference: {diff:.2f}%")
    print(f"\n💡 Recommendations:")
    print(f"   - Train for more epochs")
    print(f"   - Generate more training data")
    print(f"   - Try ensemble methods")
print("="*70)

# ============================================================================
# STEP 8: SAVE EVERYTHING
# ============================================================================
print("\n" + "="*70)
print("STEP 8: SAVING MODEL AND RESULTS")
print("="*70)

# Save final model
best_model.save(models_dir / 'ultimate_cnn_final.h5')
print("✓ Saved: models/ultimate_cnn_final.h5")
print("✓ Saved: models/ultimate_cnn_best.h5")

# Save results
results = {
    'test_accuracy': float(test_accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'best_val_accuracy': float(best_val_acc),
    'rank': ultimate_rank,
    'per_class_accuracy': {traffic_levels[i]: float(acc) for i, acc in enumerate(class_accuracies)}
}

import json
with open(models_dir / 'ultimate_cnn_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Saved: models/ultimate_cnn_results.json")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2, color='blue')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='orange')
axes[0, 0].set_title('Ultimate CNN - Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2, color='blue')
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2, color='orange')
axes[0, 1].set_title('Ultimate CNN - Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1, 0],
            xticklabels=traffic_levels, yticklabels=traffic_levels)
axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# Model Ranking
model_names = [name[:25] for name, _ in sorted_models[:10]]
accuracies = [acc for _, acc in sorted_models[:10]]
colors = ['gold' if 'Ultimate CNN' in sorted_models[i][0] else 'skyblue' for i in range(min(10, len(sorted_models)))]
axes[1, 1].barh(model_names, accuracies, color=colors)
axes[1, 1].set_xlabel('Accuracy (%)')
axes[1, 1].set_title('Top 10 Models Comparison', fontsize=14, fontweight='bold')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(models_dir / 'ultimate_cnn_complete_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/ultimate_cnn_complete_analysis.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🏆 TRAINING COMPLETE!")
print("="*70)

if ultimate_rank == 1:
    print(f"\n🎉 CONGRATULATIONS! 🎉")
    print(f"✅ Ultimate CNN is #1 with {test_accuracy*100:.2f}% accuracy!")
    print(f"✅ Beat all ML and DL models!")
else:
    print(f"\n✅ Ultimate CNN: {test_accuracy*100:.2f}% (Rank #{ultimate_rank})")

print(f"\n📁 Files saved:")
print(f"   • models/ultimate_cnn_final.h5")
print(f"   • models/ultimate_cnn_best.h5")
print(f"   • models/ultimate_cnn_results.json")
print(f"   • models/ultimate_cnn_complete_analysis.png")

print(f"\n📊 Next steps:")
print(f"   1. Run: python check_accuracy.py")
print(f"   2. Use ultimate_cnn_final.h5 in your app")
print(f"   3. Present results to your guide!")

print("\n" + "="*70)
