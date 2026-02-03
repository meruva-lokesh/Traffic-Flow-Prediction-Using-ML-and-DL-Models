"""
Generate perfect CNN architecture diagram with proper spacing and alignment
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs('models', exist_ok=True)

# Create figure with blue background
fig, ax = plt.subplots(figsize=(7, 13), facecolor='#1e3a8a')
ax.set_xlim(0, 7)
ax.set_ylim(-3, 13)
ax.axis('off')
ax.set_facecolor('#1e3a8a')

# Title
ax.text(3.5, 12.5, 'Proposed 1D CNN', ha='center', va='center', 
        fontsize=20, fontweight='bold', color='white')
ax.text(3.5, 12.0, 'Architecture', ha='center', va='center', 
        fontsize=20, fontweight='bold', color='white')

# Input
box = FancyBboxPatch((1.5, 10.8), 4, 0.8, boxstyle="round,pad=0.1", 
                     facecolor='#9333ea', edgecolor='white', linewidth=3)
ax.add_patch(box)
ax.text(3.5, 11.2, '17 Features', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')

# Arrow
arrow = FancyArrowPatch((3.5, 10.6), (3.5, 10.1), arrowstyle='->', 
                       mutation_scale=30, linewidth=5, color='#f97316')
ax.add_patch(arrow)

# CONV BLOCK 1
box = FancyBboxPatch((0.8, 8.5), 5.4, 1.5, boxstyle="round,pad=0.1", 
                     facecolor='#ea580c', edgecolor='white', linewidth=3)
ax.add_patch(box)
ax.text(3.5, 9.6, 'CONV BLOCK 1', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white')
ax.text(3.5, 9.2, '128 filters, kernel=5', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(3.5, 8.8, '→BN→ReLU→MaxPool→Drop(0.25)', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')

arrow = FancyArrowPatch((3.5, 8.3), (3.5, 7.8), arrowstyle='->', 
                       mutation_scale=30, linewidth=5, color='#f97316')
ax.add_patch(arrow)

# CONV BLOCK 2
box = FancyBboxPatch((0.8, 6.2), 5.4, 1.5, boxstyle="round,pad=0.1", 
                     facecolor='#dc2626', edgecolor='white', linewidth=3)
ax.add_patch(box)
ax.text(3.5, 7.3, 'CONV BLOCK 2', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white')
ax.text(3.5, 6.9, '256 filters, kernel=3', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(3.5, 6.5, '→BN→ReLU→MaxPool→Drop(0.30)', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')

arrow = FancyArrowPatch((3.5, 6.0), (3.5, 5.5), arrowstyle='->', 
                       mutation_scale=30, linewidth=5, color='#f97316')
ax.add_patch(arrow)

# CONV BLOCK 3
box = FancyBboxPatch((0.8, 3.9), 5.4, 1.5, boxstyle="round,pad=0.1", 
                     facecolor='#059669', edgecolor='white', linewidth=3)
ax.add_patch(box)
ax.text(3.5, 5.0, 'CONV BLOCK 3', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white')
ax.text(3.5, 4.6, '384 filters, kernel=3', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(3.5, 4.2, '→BN→ReLU→MaxPool→Drop(0.35)', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')

arrow = FancyArrowPatch((3.5, 3.7), (3.5, 3.2), arrowstyle='->', 
                       mutation_scale=30, linewidth=5, color='#f97316')
ax.add_patch(arrow)

# CONV BLOCK 4
box = FancyBboxPatch((0.8, 1.6), 5.4, 1.5, boxstyle="round,pad=0.1", 
                     facecolor='#9333ea', edgecolor='white', linewidth=3)
ax.add_patch(box)
ax.text(3.5, 2.7, 'CONV BLOCK 4', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white')
ax.text(3.5, 2.3, '512 filters, kernel=3', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(3.5, 1.9, '→BN→ReLU→MaxPool→Drop(0.40)', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')

arrow = FancyArrowPatch((3.5, 1.4), (3.5, 1.0), arrowstyle='->', 
                       mutation_scale=30, linewidth=5, color='#f97316')
ax.add_patch(arrow)

# GLOBAL AVG POOL
box = FancyBboxPatch((1.3, 0.2), 4.4, 0.7, boxstyle="round,pad=0.1", 
                     facecolor='#65a30d', edgecolor='white', linewidth=3)
ax.add_patch(box)
ax.text(3.5, 0.55, 'GLOBAL AVG POOL', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white')

arrow = FancyArrowPatch((3.5, 0.0), (3.5, -0.4), arrowstyle='->', 
                       mutation_scale=30, linewidth=5, color='#f97316')
ax.add_patch(arrow)

# DENSE 768
box = FancyBboxPatch((1.0, -1.2), 5.0, 0.7, boxstyle="round,pad=0.1", 
                     facecolor='#f59e0b', edgecolor='white', linewidth=3)
ax.add_patch(box)
ax.text(3.5, -0.85, 'DENSE 768', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white')
ax.text(3.5, -1.1, 'BN→ReLU→Drop(0.40)', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')

arrow = FancyArrowPatch((3.5, -1.4), (3.5, -1.7), arrowstyle='->', 
                       mutation_scale=30, linewidth=5, color='#f97316')
ax.add_patch(arrow)

# DENSE 384
box = FancyBboxPatch((1.0, -2.5), 5.0, 0.7, boxstyle="round,pad=0.1", 
                     facecolor='#0ea5e9', edgecolor='white', linewidth=3)
ax.add_patch(box)
ax.text(3.5, -2.15, 'DENSE 384', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white')
ax.text(3.5, -2.4, 'BN→ReLU→Drop(0.35)', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')

arrow = FancyArrowPatch((3.5, -2.7), (3.5, -3.0), arrowstyle='->', 
                       mutation_scale=30, linewidth=5, color='#f97316')
ax.add_patch(arrow)

# DENSE 192
box = FancyBboxPatch((1.0, -3.8), 5.0, 0.7, boxstyle="round,pad=0.1", 
                     facecolor='#0ea5e9', edgecolor='white', linewidth=3)
ax.add_patch(box)
ax.text(3.5, -3.45, 'DENSE 192', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white')
ax.text(3.5, -3.7, 'BN→ReLU→Drop(0.30)', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')

arrow = FancyArrowPatch((3.5, -4.0), (3.5, -4.3), arrowstyle='->', 
                       mutation_scale=30, linewidth=5, color='#f97316')
ax.add_patch(arrow)

# OUTPUT
box = FancyBboxPatch((0.5, -5.5), 6.0, 1.1, boxstyle="round,pad=0.1", 
                     facecolor='#dc2626', edgecolor='white', linewidth=3)
ax.add_patch(box)
ax.text(3.5, -4.8, 'OUTPUT', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')
ax.text(3.5, -5.1, 'Dense → Softmax (4 classes)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.text(3.5, -5.35, 'Low | Medium | High | Severe', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white')

arrow = FancyArrowPatch((3.5, -5.7), (3.5, -6.0), arrowstyle='->', 
                       mutation_scale=30, linewidth=5, color='#f97316')
ax.add_patch(arrow)

# Parameters box (wider to fit all text)
box = FancyBboxPatch((0.2, -7.1), 6.6, 0.9, boxstyle="round,pad=0.15", 
                     facecolor='#fbbf24', edgecolor='#92400e', linewidth=3)
ax.add_patch(box)
ax.text(3.5, -6.45, 'Total Parameters: ~2.5M | Training: ~22 min/fold', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')
ax.text(3.5, -6.8, 'Inference: 8ms', ha='center', va='center',
        fontsize=11, fontweight='bold', color='black')

arrow = FancyArrowPatch((3.5, -7.3), (3.5, -7.6), arrowstyle='->', 
                       mutation_scale=30, linewidth=5, color='#f97316')
ax.add_patch(arrow)

# Optimizer box (wider to fit all text)
box = FancyBboxPatch((0.4, -8.6), 6.2, 0.8, boxstyle="round,pad=0.1", 
                     facecolor='#374151', edgecolor='white', linewidth=3)
ax.add_patch(box)
ax.text(3.5, -8.0, 'OPTIMIZER: Adam | LOSS: Categorical Crossentropy', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#fbbf24')

plt.tight_layout()
plt.savefig('models/cnn_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='#1e3a8a', pad_inches=0.1)
print("✅ Perfect CNN architecture diagram created!")
print("📁 Saved: models/cnn_architecture.png")
plt.close()
