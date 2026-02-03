"""
Generate colorful CNN architecture diagram matching reference style
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs('models', exist_ok=True)

# Create figure with blue background (optimized for IEEE single column)
fig, ax = plt.subplots(figsize=(5, 10), facecolor='#1e3a8a')
ax.set_xlim(0, 5)
ax.set_ylim(-3.0, 11.5)
ax.axis('off')
ax.set_facecolor('#1e3a8a')

# Architecture layers with colors matching reference (optimized for IEEE column)
layers = [
    ("Proposed 1D CNN\nArchitecture", "", "none", 11.0, 0, 14),
    ("17 Features", "INPUT", "#9333ea", 10.2, 0.8, 11),
    ("CONV BLOCK 1\n128 filters, kernel=5\n→BN→ReLU→MaxPool→Drop(0.25)", "", "#ea580c", 9.0, 1.3, 11),
    ("CONV BLOCK 2\n256 filters, kernel=3\n→BN→ReLU→MaxPool→Drop(0.30)", "", "#dc2626", 7.6, 1.3, 11),
    ("CONV BLOCK 3\n384 filters, kernel=3\n→BN→ReLU→MaxPool→Drop(0.35)", "", "#059669", 6.2, 1.3, 11),
    ("CONV BLOCK 4\n512 filters, kernel=3\n→BN→ReLU→MaxPool→Drop(0.40)", "", "#9333ea", 4.8, 1.3, 11),
    ("GLOBAL AVG POOL", "", "#65a30d", 3.6, 1.1, 10),
    ("DENSE 768\nBN→ReLU→Drop(0.40)", "", "#f59e0b", 2.6, 1.3, 11),
    ("DENSE 384\nBN→ReLU→Drop(0.35)", "", "#0ea5e9", 1.6, 1.3, 11),
    ("DENSE 192\nBN→ReLU→Drop(0.30)", "", "#0ea5e9", 0.6, 1.3, 11),
]

# Draw layers
for i, (text, label, color, y_pos, width_offset, text_width) in enumerate(layers):
    if i == 0:
        # Title
        ax.text(2.5, y_pos, text, ha='center', va='center', 
                fontsize=13, fontweight='bold', color='white')
    else:
        # Draw box
        box = FancyBboxPatch((2.5 - width_offset, y_pos - 0.3), width_offset * 2, 0.6,
                            boxstyle="round,pad=0.08", 
                            facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(box)
        
        # Add text
        ax.text(2.5, y_pos, text, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
        
        # Draw arrow to next layer
        if i < len(layers) - 1:
            arrow = FancyArrowPatch((2.5, y_pos - 0.4), (2.5, y_pos - 0.7),
                                   arrowstyle='->', mutation_scale=20, 
                                   linewidth=3, color='#f97316')
            ax.add_patch(arrow)

# Add OUTPUT box at bottom
output_box = FancyBboxPatch((1.0, -0.9), 4, 0.7,
                           boxstyle="round,pad=0.1",
                           facecolor='#dc2626', edgecolor='white', linewidth=2.5)
ax.add_patch(output_box)
ax.text(3, -0.55, "OUTPUT\nDense → Softmax (4 classes)\nLow | Medium | High | Severe", 
        ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Arrow to output
arrow = FancyArrowPatch((3, 0.1), (3, -0.3),
                       arrowstyle='->', mutation_scale=25, 
                       linewidth=3, color='#f97316')
ax.add_patch(arrow)

# Add parameter info at bottom (wider box for full text)
param_box = FancyBboxPatch((0.8, -2.0), 6.4, 0.6,
                          boxstyle="round,pad=0.1",
                          facecolor='#fbbf24', edgecolor='#92400e', linewidth=3)
ax.add_patch(param_box)
ax.text(4, -1.7, "Total Parameters: ~2.5M | Training: ~22 min/fold | Inference: 8ms",
        ha='center', va='center', fontsize=9, fontweight='bold', color='black')

# Add optimizer info (wider box)
opt_box = FancyBboxPatch((1.0, -2.8), 6.0, 0.5,
                        boxstyle="round,pad=0.1",
                        facecolor='#374151', edgecolor='white', linewidth=2)
ax.add_patch(opt_box)
ax.text(4, -2.55, "OPTIMIZER: Adam | LOSS: Categorical Crossentropy",
        ha='center', va='center', fontsize=9, fontweight='bold', color='#fbbf24')

plt.tight_layout()
plt.savefig('models/cnn_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='#1e3a8a', edgecolor='none', pad_inches=0.2)
print("✓ Generated colorful CNN architecture: models/cnn_architecture.png")
plt.close()

print("\n✅ Colorful architecture diagram created successfully!")
print("📁 File saved: models/cnn_architecture.png")
print("\n✨ Features:")
print("  - Blue background with colored blocks")
print("  - Bold white text")
print("  - Orange arrows")
print("  - Correct information (17 features, 768/384/192 dense, 4 classes)")
