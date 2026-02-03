"""
MASTER SCRIPT: Run All Models and Compare
- Standard ML Models
- Optimized 1D CNN
- Novel Attention CNN
"""

import subprocess
import pandas as pd
from pathlib import Path
import sys

print("="*80)
print("🚀 COMPLETE MODEL COMPARISON FOR PUBLICATION")
print("="*80)

print("\n📋 This will run:")
print("   1. Standard ML + Optimized CNN (5-fold CV)")
print("   2. Novel Attention CNN (5-fold CV)")
print("   3. Generate comparison table")

input("\nPress ENTER to start (this will take 20-30 minutes)...")

# Run main training
print("\n" + "="*80)
print("STEP 1: Running Standard Models + Optimized CNN")
print("="*80)
result1 = subprocess.run([sys.executable, 'train_stable_publication.py'], capture_output=False)

if result1.returncode != 0:
    print("❌ Error in main training script")
    sys.exit(1)

# Run attention CNN
print("\n" + "="*80)
print("STEP 2: Running Novel Attention CNN")
print("="*80)
result2 = subprocess.run([sys.executable, 'train_attention_cnn.py'], capture_output=False)

if result2.returncode != 0:
    print("❌ Error in attention CNN script")
    sys.exit(1)

# Load and combine results
print("\n" + "="*80)
print("STEP 3: Generating Final Comparison Table")
print("="*80)

results_dir = Path('publication_results')

# Load standard results
df_standard = pd.read_csv(results_dir / 'stable_results_5fold.csv')

# Load attention results
df_attention = pd.read_csv(results_dir / 'attention_cnn_results.csv')

# Combine
print("\n📊 COMPLETE RESULTS (Ranked by Accuracy):")
print("="*80)
print(df_standard.to_string(index=False))
print("\nNOVEL ARCHITECTURE:")
print("-"*80)
print(df_attention.to_string(index=False))

# Determine winner
best_standard = df_standard.iloc[0]
attention_acc = df_attention.iloc[0]['Accuracy Mean (%)']

print("\n" + "="*80)
print("🏆 PUBLICATION RESULTS")
print("="*80)
print(f"\nBest Traditional Model: {best_standard['Model']}")
print(f"  Accuracy: {best_standard['Accuracy Mean (%)']:.2f}% ± {best_standard['Accuracy Std (%)']:.2f}%")
print(f"\nOptimized 1D CNN:")
cnn_row = df_standard[df_standard['Model'] == '1D CNN'].iloc[0]
print(f"  Accuracy: {cnn_row['Accuracy Mean (%)']:.2f}% ± {cnn_row['Accuracy Std (%)']:.2f}%")
print(f"\nNovel Attention CNN:")
print(f"  Accuracy: {attention_acc:.2f}% ± {df_attention.iloc[0]['Accuracy Std (%)']:.2f}%")

# Save combined results
combined_df = pd.concat([df_standard, df_attention], ignore_index=True)
combined_df = combined_df.sort_values('Accuracy Mean (%)', ascending=False).reset_index(drop=True)
combined_df.insert(0, 'Rank', range(1, len(combined_df) + 1))
combined_df.to_csv(results_dir / 'COMPLETE_COMPARISON.csv', index=False)

print(f"\n✅ Complete comparison saved: {results_dir}/COMPLETE_COMPARISON.csv")
print("\n💡 Use these results in your conference paper!")
