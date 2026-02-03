"""
Script to Check Accuracy of All Models (ML + DL)
Run: python check_accuracy.py
"""

import joblib
import pandas as pd
import json
from pathlib import Path

print("="*70)
print("MODEL ACCURACY CHECKER")
print("="*70)

models_dir = Path("models")

# ============================================================================
# PART 1: TRADITIONAL ML MODELS
# ============================================================================
print("\n" + "="*70)
print("TRADITIONAL ML MODELS (5 Models)")
print("="*70)

try:
    # Load comparison results
    comparison_df = joblib.load(models_dir / 'model_comparison.pkl')
    
    print("\n📊 ML Model Performance:")
    print("-" * 70)
    
    # Display in a nice table
    print(f"{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    
    for idx, row in comparison_df.iterrows():
        print(f"{row['Model']:<30} {row['Accuracy']*100:>10.2f}% {row['Precision']*100:>10.2f}% "
              f"{row['Recall']*100:>10.2f}% {row['F1-Score']*100:>10.2f}%")
    
    print("\n🏆 BEST ML MODEL:")
    best_ml = comparison_df.iloc[0]
    print(f"   Model: {best_ml['Model']}")
    print(f"   Accuracy: {best_ml['Accuracy']*100:.2f}%")
    print(f"   F1-Score: {best_ml['F1-Score']*100:.2f}%")
    
except FileNotFoundError:
    print("\n❌ ML model results not found!")
    print("   Run: python src/train_all_models.py")

# ============================================================================
# PART 2: DEEP LEARNING MODELS
# ============================================================================
print("\n" + "="*70)
print("DEEP LEARNING MODELS (4 Models)")
print("="*70)

try:
    # Check if CSV exists
    if (models_dir / 'deep_learning_comparison.csv').exists():
        dl_df = pd.read_csv(models_dir / 'deep_learning_comparison.csv')
        
        print("\n📊 DL Model Performance:")
        print("-" * 70)
        print(f"{'Model':<20} {'Test Accuracy':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        
        for idx, row in dl_df.iterrows():
            print(f"{row['Model']:<20} {row['Test Accuracy (%)']:>13.2f}% "
                  f"{row['Precision (%)']:>10.2f}% {row['Recall (%)']:>10.2f}% {row['F1-Score (%)']:>10.2f}%")
        
        # Find best DL model
        best_idx = dl_df['Test Accuracy (%)'].idxmax()
        best_dl = dl_df.iloc[best_idx]
        
        print("\n🏆 BEST DL MODEL:")
        print(f"   Model: {best_dl['Model']}")
        print(f"   Accuracy: {best_dl['Test Accuracy (%)']:.2f}%")
        print(f"   F1-Score: {best_dl['F1-Score (%)']:.2f}%")
        print(f"   Training Time: {best_dl['Training Time (s)']:.2f} seconds")
        
    elif (models_dir / 'deep_learning_results.json').exists():
        with open(models_dir / 'deep_learning_results.json', 'r') as f:
            dl_results = json.load(f)
        
        print("\n📊 DL Model Performance:")
        print("-" * 70)
        print(f"{'Model':<20} {'Test Accuracy':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        
        for model_name, result in dl_results.items():
            print(f"{model_name:<20} {result['test_accuracy']*100:>13.2f}% "
                  f"{result['precision']*100:>10.2f}% {result['recall']*100:>10.2f}% "
                  f"{result['f1_score']*100:>10.2f}%")
        
        # Find best
        best_model = max(dl_results.items(), key=lambda x: x[1]['test_accuracy'])
        print("\n🏆 BEST DL MODEL:")
        print(f"   Model: {best_model[0]}")
        print(f"   Accuracy: {best_model[1]['test_accuracy']*100:.2f}%")
        print(f"   F1-Score: {best_model[1]['f1_score']*100:.2f}%")
    else:
        print("\n❌ DL model results not found!")
        print("   Run: python src/train_deep_learning_models.py")
        
except Exception as e:
    print(f"\n❌ Error loading DL results: {e}")
    print("   Run: python src/train_deep_learning_models.py")

# ============================================================================
# PART 3: OVERALL COMPARISON
# ============================================================================
print("\n" + "="*70)
print("OVERALL COMPARISON (ALL 9 MODELS)")
print("="*70)

try:
    all_models = []
    
    # Add ML models
    if 'comparison_df' in locals():
        for idx, row in comparison_df.iterrows():
            all_models.append({
                'Model': f"{row['Model']} (ML)",
                'Accuracy': row['Accuracy'] * 100,
                'Type': 'Traditional ML'
            })
    
    # Add DL models
    if 'dl_df' in locals():
        for idx, row in dl_df.iterrows():
            all_models.append({
                'Model': f"{row['Model']} (DL)",
                'Accuracy': row['Test Accuracy (%)'],
                'Type': 'Deep Learning'
            })
    elif 'dl_results' in locals():
        for model_name, result in dl_results.items():
            all_models.append({
                'Model': f"{model_name} (DL)",
                'Accuracy': result['test_accuracy'] * 100,
                'Type': 'Deep Learning'
            })
    
    if all_models:
        # Sort by accuracy
        all_models_sorted = sorted(all_models, key=lambda x: x['Accuracy'], reverse=True)
        
        print("\n📊 All Models Ranked by Accuracy:")
        print("-" * 70)
        print(f"{'Rank':<6} {'Model':<40} {'Accuracy':<12} {'Type':<20}")
        print("-" * 70)
        
        for rank, model in enumerate(all_models_sorted, 1):
            symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{rank:<6} {model['Model']:<40} {model['Accuracy']:>10.2f}% {model['Type']:<20} {symbol}")
        
        print("\n" + "="*70)
        print("🏆 OVERALL BEST MODEL:")
        best = all_models_sorted[0]
        print(f"   {best['Model']}")
        print(f"   Accuracy: {best['Accuracy']:.2f}%")
        print(f"   Category: {best['Type']}")
        print("="*70)
        
except Exception as e:
    print(f"\n❌ Could not create overall comparison: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY & NEXT STEPS")
print("="*70)

print("\n✅ To view results:")
print("   • ML Models: Check models/model_comparison.pkl")
print("   • DL Models: Check models/deep_learning_comparison.csv")
print("   • Visualizations: models/training_history.png, confusion_matrices_dl.png")

print("\n✅ To retrain models:")
print("   • ML: python src/train_all_models.py")
print("   • DL: python src/train_deep_learning_models.py")

print("\n✅ To run predictions:")
print("   • streamlit run app.py")

print("\n" + "="*70)
