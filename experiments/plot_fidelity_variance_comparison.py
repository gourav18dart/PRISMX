"""
Script: generate_fidelity_variance_comparison.py
Description: Generates a clear, publication-quality comparative figure illustrating:
1. Classification probabilities (Full Model 512D vs 3 PCs vs 15 PCs)
2. Cumulative variance captured & information lost
3. Feature reconstruction error
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Output directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, "fidelity_and_variance_comparison.png")

# Data from experiments
categories = ["Original Model\n(All 512 Dims)", "Standard PRISM\n(Only 3 PCs)", "High-Variance PRISMX\n(15 PCs)"]
coyote_probs = [0.9255, 0.4740, 0.9020]
wolf_probs = [0.9118, 0.6035, 0.9017]

variance_captured = [100.0, 56.5, 86.7]
variance_lost = [0.0, 43.5, 13.3]
recon_error = [0.0, 59.2, 32.8]

# Setup plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(15, 6), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.32)

# ==========================================
# SUBPLOT 1: Prediction Probability (Fidelity)
# ==========================================
ax1 = fig.add_subplot(gs[0])
x = np.arange(len(categories))
width = 0.32

rects1 = ax1.bar(x - width/2, coyote_probs, width, label='Coyote Class', color='#2b5c8f', edgecolor='black', linewidth=1)
rects2 = ax1.bar(x + width/2, wolf_probs, width, label='Timber Wolf Class', color='#d95f02', edgecolor='black', linewidth=1)

ax1.set_ylabel('Softmax Prediction Confidence', fontsize=12, fontweight='bold')
ax1.set_title('Prediction Confidence (Model Fidelity)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax1.set_ylim(0, 1.15)
ax1.axhline(0.90, color='gray', linestyle=':', alpha=0.7, label='Original Baseline (~0.91-0.93)')
ax1.legend(loc='upper right', frameon=True, fontsize=9)

# Value annotations
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}\n({h*100:.1f}%)',
                 xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1a3b5c')

for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}\n({h*100:.1f}%)',
                 xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#8c3a00')

# Highlight the drop and restoration
ax1.annotate('Severe Collapse!\n(-45% confidence)', xy=(1, 0.48), xytext=(1, 0.72),
             arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
             ha='center', fontsize=9, fontweight='bold', color='red',
             bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="red", lw=1))

ax1.annotate('Full Fidelity Restored!\n(Back to >0.90)', xy=(2, 0.91), xytext=(2, 1.04),
             arrowprops=dict(facecolor='green', shrink=0.08, width=1.5, headwidth=6),
             ha='center', fontsize=9, fontweight='bold', color='darkgreen',
             bbox=dict(boxstyle="round,pad=0.3", fc="#e6ffe6", ec="darkgreen", lw=1))

# ==========================================
# SUBPLOT 2: Variance Captured vs Lost
# ==========================================
ax2 = fig.add_subplot(gs[1])
p1 = ax2.bar(categories, variance_captured, width=0.55, label='Captured Variance (Retained Info)', color='#1b9e77', edgecolor='black', linewidth=1)
p2 = ax2.bar(categories, variance_lost, width=0.55, bottom=variance_captured, label='Lost Variance (Discarded Info)', color='#d9534f', edgecolor='black', linewidth=1)

ax2.set_ylabel('Percentage of Information (%)', fontsize=12, fontweight='bold')
ax2.set_title('Feature Variance (Information Kept vs Lost)', fontsize=13, fontweight='bold', pad=12)
ax2.set_ylim(0, 115)
ax2.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax2.legend(loc='upper right', frameon=True, fontsize=9)

for i, total in enumerate(variance_captured):
    ax2.text(i, total / 2, f'{total:.1f}% Kept', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    if variance_lost[i] > 0:
        ax2.text(i, total + (variance_lost[i] / 2), f'{variance_lost[i]:.1f}% Lost', ha='center', va='center', color='white', fontweight='bold', fontsize=10)

# ==========================================
# SUBPLOT 3: Relative Feature Reconstruction Error
# ==========================================
ax3 = fig.add_subplot(gs[2])
colors = ['#4daf4a', '#e41a1c', '#377eb8']
bars = ax3.bar(categories, recon_error, width=0.55, color=colors, edgecolor='black', linewidth=1)

ax3.set_ylabel('Relative Feature Reconstruction Error (%)', fontsize=12, fontweight='bold')
ax3.set_title('Feature Reconstruction Error\n||F - F_recon|| / ||F||', fontsize=13, fontweight='bold', pad=12)
ax3.set_ylim(0, 75)
ax3.set_xticklabels(categories, fontsize=10, fontweight='bold')

for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}% Error', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax3.text(1, 63, 'Large distortion\nfrom missing PCs', ha='center', fontsize=9, color='#e41a1c', fontweight='bold')
ax3.text(2, 36, 'Error cut by almost half!', ha='center', fontsize=9, color='#377eb8', fontweight='bold')

plt.suptitle('Comparison: Original CNN (512 Dims) vs. Standard PRISM (3 PCs) vs. High-Variance PRISMX (15 PCs)',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Figure successfully saved to: {save_path}")
