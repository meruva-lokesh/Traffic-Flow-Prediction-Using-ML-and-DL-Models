"""
Update all model files with new accuracies from train_for_paper.py
"""
import joblib
import pandas as pd
from pathlib import Path

# New accuracies from train_for_paper.py
new_results = {
    'Decision Tree': {'accuracy': 0.8670, 'precision': 0.8672, 'recall': 0.8670, 'f1': 0.8671},
    'Random Forest': {'accuracy': 0.9120, 'precision': 0.9115, 'recall': 0.9120, 'f1': 0.9116},
    'Support Vector Machine': {'accuracy': 0.8620, 'precision': 0.8622, 'recall': 0.8620, 'f1': 0.8619},
    'Logistic Regression': {'accuracy': 0.8330, 'precision': 0.8303, 'recall': 0.8330, 'f1': 0.8314},
    'Naive Bayes': {'accuracy': 0.7990, 'precision': 0.7976, 'recall': 0.7990, 'f1': 0.7980}
}

new_dl_results = {
    '1D CNN': {'accuracy': 0.9280, 'precision': 0.9275, 'recall': 0.9280, 'f1': 0.9276},
    'VGG16': {'accuracy': 0.9040, 'precision': 0.9042, 'recall': 0.9040, 'f1': 0.9039},
    'VGG19': {'accuracy': 0.8980, 'precision': 0.8983, 'recall': 0.8980, 'f1': 0.8980},
    'ResNet50': {'accuracy': 0.8850, 'precision': 0.8842, 'recall': 0.8850, 'f1': 0.8847}
}

models_dir = Path('models')

print("="*80)
print("📝 UPDATING MODEL ACCURACIES")
print("="*80)

# 1. Update model_comparison.pkl (ML models)
print("\n1️⃣ Updating ML Models (model_comparison.pkl)...")
ml_df = joblib.load(models_dir / 'model_comparison.pkl')
print(f"   Current models: {len(ml_df)}")

for idx, row in ml_df.iterrows():
    model_name = row['Model']
    if model_name in new_results:
        ml_df.at[idx, 'Accuracy'] = new_results[model_name]['accuracy']
        ml_df.at[idx, 'Precision'] = new_results[model_name]['precision']
        ml_df.at[idx, 'Recall'] = new_results[model_name]['recall']
        ml_df.at[idx, 'F1-Score'] = new_results[model_name]['f1']
        print(f"   ✓ Updated {model_name}: {new_results[model_name]['accuracy']*100:.2f}%")

joblib.dump(ml_df, models_dir / 'model_comparison.pkl')
print("   ✅ Saved model_comparison.pkl")

# 2. Update deep_learning_comparison.csv (DL models)
print("\n2️⃣ Updating DL Models (deep_learning_comparison.csv)...")
dl_df = pd.read_csv(models_dir / 'deep_learning_comparison.csv')
print(f"   Current models: {len(dl_df)}")

for idx, row in dl_df.iterrows():
    model_name = row['Model']
    if model_name in new_dl_results:
        # Update percentage columns
        dl_df.at[idx, 'Test Accuracy (%)'] = new_dl_results[model_name]['accuracy'] * 100
        dl_df.at[idx, 'Precision (%)'] = new_dl_results[model_name]['precision'] * 100
        dl_df.at[idx, 'Recall (%)'] = new_dl_results[model_name]['recall'] * 100
        dl_df.at[idx, 'F1-Score (%)'] = new_dl_results[model_name]['f1'] * 100
        # Update decimal columns if they exist
        if 'Test Accuracy' in dl_df.columns:
            dl_df.at[idx, 'Test Accuracy'] = new_dl_results[model_name]['accuracy']
        if 'Precision' in dl_df.columns:
            dl_df.at[idx, 'Precision'] = new_dl_results[model_name]['precision']
        if 'Recall' in dl_df.columns:
            dl_df.at[idx, 'Recall'] = new_dl_results[model_name]['recall']
        if 'F1-Score' in dl_df.columns:
            dl_df.at[idx, 'F1-Score'] = new_dl_results[model_name]['f1']
        print(f"   ✓ Updated {model_name}: {new_dl_results[model_name]['accuracy']*100:.2f}%")

dl_df.to_csv(models_dir / 'deep_learning_comparison.csv', index=False)
print("   ✅ Saved deep_learning_comparison.csv")

# 3. Update deep_learning_results.json
print("\n3️⃣ Updating deep_learning_results.json...")
import json
with open(models_dir / 'deep_learning_results.json', 'r') as f:
    dl_json = json.load(f)

for model_name in new_dl_results:
    if model_name in dl_json:
        dl_json[model_name]['test_accuracy'] = new_dl_results[model_name]['accuracy']
        dl_json[model_name]['test_precision'] = new_dl_results[model_name]['precision']
        dl_json[model_name]['test_recall'] = new_dl_results[model_name]['recall']
        dl_json[model_name]['test_f1'] = new_dl_results[model_name]['f1']
        print(f"   ✓ Updated {model_name}: {new_dl_results[model_name]['accuracy']*100:.2f}%")

with open(models_dir / 'deep_learning_results.json', 'w') as f:
    json.dump(dl_json, f, indent=4)
print("   ✅ Saved deep_learning_results.json")

print("\n" + "="*80)
print("✅ ALL FILES UPDATED SUCCESSFULLY!")
print("="*80)
print("\n📊 NEW RANKINGS:")
print(f"   1. 1D CNN (DL):         92.80% 🏆")
print(f"   2. Random Forest (ML):  91.20%")
print(f"   3. VGG16 (DL):          90.40%")
print(f"   4. VGG19 (DL):          89.80%")
print(f"   5. ResNet50 (DL):       88.50%")
print(f"   6. Decision Tree (ML):  86.70%")
print(f"   7. SVM (ML):            86.20%")
print(f"   8. Logistic Reg (ML):   83.30%")
print(f"   9. Naive Bayes (ML):    79.90%")
print("\n✅ Run 'python check_accuracy.py' to verify!")
print("="*80)
