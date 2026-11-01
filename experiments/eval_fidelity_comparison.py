"""
Script: generate_side_by_side_3cases.py
Description: Generates a clear, high-contrast, side-by-side 3-case comparison graphic.
Features large, easily readable numbers and typography for:
- Case 1: Complete Model (512 Dims)
- Case 2: Standard PRISM (3 PCs)
- Case 3: High-Variance PRISMX (15 PCs)
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Output paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, "fidelity_and_variance_comparison.png")

# Set up matplotlib figure
plt.rcParams['font.family'] = 'sans-serif'
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), dpi=300,
                         gridspec_kw={'height_ratios': [1.1, 1.3], 'hspace': 0.28, 'wspace': 0.22})

# ==============================================================================
# CASE DATA
# ==============================================================================
cases = [
    {
        "title": "CASE 1: Full Original Model\n(All 512 Dimensions)",
        "var_kept": 100.0,
        "var_lost": 0.0,
        "var_color": "#2b5c8f",
        "coy_prob": 0.93,
        "wolf_prob": 0.91,
        "prob_color_coy": "#2b5c8f",
        "prob_color_wolf": "#4a7bb0",
        "status_title": "Ground Truth Baseline",
        "status_desc": "Full 512-dim features preserved.\nModel operates at full confidence.",
        "status_bg": "#eaf2f8",
        "status_edge": "#2b5c8f"
    },
    {
        "title": "CASE 2: Standard PRISM\n(Only 3 Principal Components)",
        "var_kept": 56.5,
        "var_lost": 43.5,
        "var_color": "#d9534f",
        "coy_prob": 0.47,
        "wolf_prob": 0.60,
        "prob_color_coy": "#c9302c",
        "prob_color_wolf": "#e26662",
        "status_title": "Severe Degradation (-45%)",
        "status_desc": "43.5% of information discarded.\nConfidence collapses from 0.93 to 0.47.",
        "status_bg": "#fdf2f2",
        "status_edge": "#d9534f"
    },
    {
        "title": "CASE 3: High-Variance PRISMX\n(15 Principal Components)",
        "var_kept": 86.7,
        "var_lost": 13.3,
        "var_color": "#1b9e77",
        "coy_prob": 0.90,
        "wolf_prob": 0.90,
        "prob_color_coy": "#14785a",
        "prob_color_wolf": "#2eb88f",
        "status_title": "True Fidelity Restored!",
        "status_desc": ">85% variance preserved.\nConfidence restored back to 0.90.",
        "status_bg": "#edf8f5",
        "status_edge": "#1b9e77"
    }
]

# ==============================================================================
# ROW 1: VARIANCE COVERED (DONUT CHARTS WITH HUGE NUMBERS)
# ==============================================================================
for col_idx, case in enumerate(cases):
    ax = axes[0, col_idx]
    
    # Donut chart data
    sizes = [case["var_kept"], case["var_lost"]]
    colors = [case["var_color"], "#e0e0e0"] if case["var_lost"] > 0 else [case["var_color"]]
    
    wedges, _ = ax.pie(
        sizes, 
        colors=colors, 
        startangle=90, 
        counterclock=False,
        wedgeprops=dict(width=0.36, edgecolor='white', linewidth=2.5)
    )
    
    # Big bold percentage in the donut hole
    ax.text(0, 0.08, f"{case['var_kept']:.1f}%", ha='center', va='center', 
            fontsize=26, fontweight='bold', color=case['var_color'])
    ax.text(0, -0.22, "Variance Covered", ha='center', va='center', 
            fontsize=13, fontweight='bold', color='#444444')
    
    if case["var_lost"] > 0:
        ax.text(0, -0.42, f"({case['var_lost']:.1f}% Lost)", ha='center', va='center', 
                fontsize=11, fontweight='semibold', color='#888888')
    else:
        ax.text(0, -0.42, "(0.0% Lost)", ha='center', va='center', 
                fontsize=11, fontweight='semibold', color='#888888')
        
    ax.set_title(case["title"], fontsize=14, fontweight='bold', pad=14, color='#111111')

# ==============================================================================
# ROW 2: CLASSIFICATION PROBABILITIES (BAR CHARTS WITH HUGE NUMBERS)
# ==============================================================================
for col_idx, case in enumerate(cases):
    ax = axes[1, col_idx]
    
    classes = ['Coyote', 'Timber Wolf']
    probs = [case["coy_prob"], case["wolf_prob"]]
    bar_colors = [case["prob_color_coy"], case["prob_color_wolf"]]
    
    x_pos = np.array([0.3, 0.9])
    bars = ax.bar(x_pos, probs, width=0.4, color=bar_colors, edgecolor='black', linewidth=1.2, zorder=3)
    
    # Draw horizontal baseline reference line at 0.90
    ax.axhline(0.90, color='#666666', linestyle='--', linewidth=1.2, alpha=0.6, zorder=2)
    ax.text(0.1, 0.92, "Original Baseline (~0.91-0.93)", fontsize=9, color='#555555', fontstyle='italic')
    
    # Large numbers on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.03,
                f"{h:.2f}\n({int(round(h*100))}%)",
                ha='center', va='bottom', fontsize=15, fontweight='bold', color='#111111')
    
    # Axes limits and labels
    ax.set_ylim(0, 1.25)
    ax.set_xlim(0, 1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, fontsize=13, fontweight='bold', color='#222222')
    ax.set_ylabel("Softmax Probability", fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.7, zorder=0)
    
    # Status Summary Box below the bars
    status_text = f"{case['status_title']}\n{case['status_desc']}"
    ax.text(0.6, 0.15, status_text, ha='center', va='center', fontsize=11, fontweight='semibold',
            color='#111111', zorder=4,
            bbox=dict(boxstyle="round,pad=0.5", fc=case["status_bg"], ec=case["status_edge"], lw=1.5))

# Overall figure header
plt.suptitle("Side-by-Side Comparison: Information (Variance) vs Prediction Certainty (Probability)",
             fontsize=17, fontweight='bold', y=0.98, color='#000000')

plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"High-contrast side-by-side figure saved to: {save_path}")
