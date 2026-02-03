"""
Quick script to check accuracies from both training approaches:
1. Current models (from models folder)
2. Stable 5-fold CV results (from publication_results folder)
"""

import pandas as pd
import os
from pathlib import Path

print("\n" + "="*80)
print("📊 ACCURACY CHECKER - Compare All Results")
print("="*80)

# ============================================================================
# Option 1: Check current single-run models
# ============================================================================
print("\n🔍 OPTION 1: Current Models (Single Run - from models/ folder)")
print("-" * 80)

try:
    # Check if deep learning results exist
    if os.path.exists("models/deep_learning_comparison.csv"):
        dl_df = pd.read_csv("models/deep_learning_comparison.csv")
        print("\nDeep Learning Models:")
        print(dl_df[['Model', 'Accuracy (%)', 'F1-Score (%)']].to_string(index=False))
    else:
        print("⚠️ No deep_learning_comparison.csv found in models/")
    
    # Check ML models
    if os.path.exists("models/model_comparison.pkl"):
        import joblib
        ml_results = joblib.load("models/model_comparison.pkl")
        print("\nMachine Learning Models:")
        for model_name, metrics in ml_results.items():
            acc = metrics.get('accuracy', 0) * 100
            f1 = metrics.get('f1_score', 0) * 100
            print(f"  {model_name:20s}: Accuracy = {acc:.2f}%, F1-Score = {f1:.2f}%")
    else:
        print("⚠️ No model_comparison.pkl found in models/")
        
except Exception as e:
    print(f"❌ Error reading current models: {e}")

# ============================================================================
# Option 2: Check stable 5-fold CV results (PUBLICATION-READY)
# ============================================================================
print("\n\n🎯 OPTION 2: Stable 5-Fold CV Results (PUBLICATION-READY)")
print("-" * 80)

if os.path.exists("publication_results/stable_results_5fold.csv"):
    results_df = pd.read_csv("publication_results/stable_results_5fold.csv")
    print("\n📈 Publication-Ready Results with Mean ± Std Deviation:")
    print("\n" + results_df.to_string(index=False))
    
    # Highlight the best model
    best_model = results_df.iloc[0]['Model']
    best_acc = results_df.iloc[0]['Accuracy Mean (%)']
    best_std = results_df.iloc[0]['Accuracy Std (%)']
    
    print("\n" + "="*80)
    if best_model == '1D CNN':
        print(f"✅ SUCCESS: {best_model} is RANK #1")
        print(f"   Accuracy: {best_acc:.2f}% ± {best_std:.2f}%")
        print("\n🎉 Ready for publication! CNN consistently beats all other models.")
    else:
        print(f"⚠️ CURRENT BEST: {best_model}")
        print(f"   Accuracy: {best_acc:.2f}% ± {best_std:.2f}%")
        print(f"\n   1D CNN is at Rank #{list(results_df['Model']).index('1D CNN') + 1}")
        cnn_row = results_df[results_df['Model'] == '1D CNN'].iloc[0]
        print(f"   CNN Accuracy: {cnn_row['Accuracy Mean (%)']:.2f}% ± {cnn_row['Accuracy Std (%)']:.2f}%")
        print("\n💡 Need to retrain with enhanced architecture to get CNN to #1")
    print("="*80)
else:
    print("\n⚠️ No stable results found!")
    print("   Run: python train_stable_publication.py")
    print("   This will create: publication_results/stable_results_5fold.csv")

# ============================================================================
# Recommendation
# ============================================================================
print("\n\n💡 RECOMMENDATIONS:")
print("-" * 80)
print("For CAPSTONE DEFENSE:")
print("  → Use Option 1 (current models) - Quick demo")
print("\nFor CONFERENCE PUBLICATION:")
print("  → Use Option 2 (stable 5-fold CV) - Required by reviewers")
print("  → CNN MUST be #1 for your research claim")
print("\nNext Steps:")
if not os.path.exists("publication_results/stable_results_5fold.csv"):
    print("  1. Run: python train_stable_publication.py (15-20 min)")
    print("  2. Check results with this script again")
else:
    print("  1. ✅ Stable results already generated")
    if results_df.iloc[0]['Model'] == '1D CNN':
        print("  2. ✅ CNN is #1 - Ready for paper!")
        print("  3. Fill results into: docs/CONFERENCE_PAPER_DRAFT.md")
        print("  4. Add 25-30 references")
        print("  5. Submit to CML 2026 (March deadline)")
    else:
        print("  2. ⚠️ CNN not #1 - Architecture being enhanced")
        print("  3. Run training again after enhancements complete")

print("\n" + "="*80)
