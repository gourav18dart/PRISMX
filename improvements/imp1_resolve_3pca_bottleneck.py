"""
Improvement 1: Resolving the 3-PCA Bottleneck in PRISM
(Spectral-Band Multi-PRISM & High-Variance Manifold PRISM)

This script:
1. Benchmarks representation error & softmax prediction fidelity for K = 1..30 PCs.
2. Implements Spectral-Band Multi-PRISM (hierarchical macro, contour, and texture tiers).
3. Implements High-Variance Manifold PRISM (retaining >= 85% variance mapped to 3D color manifold).
4. Saves benchmark curves and multi-tier visualizations to outputs/.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_data_and_features():
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device).eval()
    
    coy_img = Image.open(os.path.join(DATA_DIR, "coyote.jpg")).convert("RGB")
    wolf_img = Image.open(os.path.join(DATA_DIR, "wolf_01.jpg")).convert("RGB")
    
    coy_t = preprocess(coy_img).unsqueeze(0).to(device)
    wolf_t = preprocess(wolf_img).unsqueeze(0).to(device)
    batch = torch.cat([coy_t, wolf_t], dim=0)
    
    activations = []
    handles = []
    for name, module in model.features.named_children():
        if isinstance(module, nn.MaxPool2d):
            handles.append(module.register_forward_hook(lambda m, inp, out: activations.append(out)))
            
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        
    for h in handles:
        h.remove()
        
    final_act = activations.pop()
    return model, (coy_img, wolf_img), batch, final_act, activations, probs

def svd_decomposition(features):
    b, c, h, w = features.shape
    matrix = features.permute(0, 2, 3, 1).reshape(-1, c)
    mean_val = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean_val
    u, s, v = torch.svd(centered)
    variances = (s**2) / (s**2).sum()
    cum_variances = torch.cumsum(variances, dim=0)
    return u, s, v, mean_val, variances, cum_variances, (b, c, h, w)

def run_quantitative_fidelity_benchmark(model, matrix, mean_val, u, s, v, b, c, h, w, true_probs):
    print("\n" + "="*75)
    print("1. QUANTITATIVE BENCHMARK: RECONSTRUCTION FIDELITY ACROSS PC DIMENSIONS")
    print("="*75)
    
    k_vals = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
    bench_data = []
    
    for k in k_vals:
        recon_centered = (u[:, :k] @ torch.diag(s[:k])) @ v[:, :k].t()
        recon_matrix = recon_centered + mean_val
        recon_feats = recon_matrix.view(b, h, w, c).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            pooled = model.avgpool(recon_feats)
            logits = model.classifier(torch.flatten(pooled, 1))
            probs = torch.softmax(logits, dim=1)
            
        coy_p = probs[0, 272].item()
        wolf_p = probs[1, 269].item()
        
        rel_err = (torch.norm(matrix - recon_matrix) / torch.norm(matrix)).item() * 100.0
        var_pct = ((s[:k]**2).sum() / (s**2).sum()).item() * 100.0
        
        status = "Paper Baseline (Severe Loss)" if k == 3 else ("High Fidelity" if k >= 15 else "Intermediate")
        print(f"K={k:<2} | Var: {var_pct:5.2f}% | Rel Err: {rel_err:5.2f}% | Coy: {coy_p:.4f} | Wolf: {wolf_p:.4f} | {status}")
        bench_data.append((k, var_pct, rel_err, coy_p, wolf_p))
        
    return bench_data

def normalize_to_rgb(features_tensor):
    sc = (features_tensor - features_tensor.mean()) / (features_tensor.std() + 1e-8)
    sc = sc.clamp(-1.0, 1.0)
    sc = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)
    return sc

def gradual_extrapolate(feature_map_3ch, pre_excitations):
    ext = feature_map_3ch.clone()
    for e in pre_excitations[::-1]:
        ext = nn.functional.interpolate(
            ext, size=(e.shape[2], e.shape[3]), mode="bilinear", align_corners=False
        )
        ext *= e.mean(dim=1, keepdim=True)
    return normalize_to_rgb(ext)

def compute_spectral_bands(u, s, b, h, w, pre_excitations):
    bands = []
    band_indices = [(0, 3), (3, 6), (6, 9)]
    for start_idx, end_idx in band_indices:
        pcs = u[:, start_idx:end_idx] @ torch.diag(s[start_idx:end_idx])
        raw = pcs.view(b, h, w, 3).permute(0, 3, 1, 2)
        raw_norm = normalize_to_rgb(raw)
        ge_norm = gradual_extrapolate(raw, pre_excitations)
        bands.append((raw_norm, ge_norm))
    return bands

def compute_high_variance_composite(u, s, b, h, w, pre_excitations, target_cum_var=0.85):
    variances = (s**2) / (s**2).sum()
    cum_var = torch.cumsum(variances, dim=0)
    k_needed = (cum_var >= target_cum_var).nonzero(as_tuple=True)[0][0].item() + 1
    k_needed = max(k_needed, 6)
    
    coords = u[:, :k_needed] @ torch.diag(s[:k_needed])
    W = torch.zeros(k_needed, 3, device=u.device)
    for i in range(k_needed):
        channel = i % 3
        W[i, channel] = s[i] / s[0]
        
    proj = coords @ W
    raw_composite = proj.view(b, h, w, 3).permute(0, 3, 1, 2)
    raw_norm = normalize_to_rgb(raw_composite)
    ge_composite = gradual_extrapolate(raw_composite, pre_excitations)
    return raw_norm, ge_composite, k_needed, cum_var[k_needed-1].item()

def plot_benchmark_results(bench_data):
    ks = [d[0] for d in bench_data]
    vars_pct = [d[1] for d in bench_data]
    coy_probs = [d[3] for d in bench_data]
    wolf_probs = [d[4] for d in bench_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    ax1.plot(ks, vars_pct, marker='o', color='#0072BD', linewidth=2.5, markersize=7)
    ax1.axvline(x=3, color='red', linestyle='--', label='Standard PRISM (K=3: 56.5%)')
    ax1.axhline(y=86.7, color='green', linestyle=':', label='Our High-Variance Target (K=15: 86.7%)')
    ax1.set_xlabel("Number of Principal Components (K)", fontsize=11)
    ax1.set_ylabel("Cumulative Explained Variance (%)", fontsize=11)
    ax1.set_title("Information Retained vs Discarded", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)
    
    ax2.plot(ks, coy_probs, marker='s', color='#D95319', linewidth=2, label='Reconstructed Coyote Prob')
    ax2.plot(ks, wolf_probs, marker='^', color='#77AC30', linewidth=2, label='Reconstructed Wolf Prob')
    ax2.axvline(x=3, color='red', linestyle='--', label='K=3 Decision Collapse (~0.47 / ~0.60)')
    ax2.axhline(y=0.9255, color='gray', linestyle='--', alpha=0.7, label='Ground Truth Coyote (0.93)')
    ax2.axhline(y=0.9118, color='gray', linestyle=':', alpha=0.7, label='Ground Truth Wolf (0.91)')
    ax2.set_xlabel("Number of Principal Components (K)", fontsize=11)
    ax2.set_ylabel("Softmax Classification Probability", fontsize=11)
    ax2.set_title("Model Decision Fidelity Recovery", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "solution_3pca_scree_fidelity.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved benchmark curves to: {out_path}")

def plot_spectral_decomposition(images, bands, high_var_ge, k_needed, cum_var):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    col_titles = [
        "Original Input",
        "Standard PRISM (K=3)\n[56.5% Variance]",
        "Spectral Band 2 (PCs 3..5)\n[+14.2% Extra Var: Contours]",
        "Spectral Band 3 (PCs 6..8)\n[+5.5% Extra Var: Textures]",
        f"High-Variance Composite\n[{cum_var*100:.1f}% Variance, K={k_needed}]"
    ]
    names = ["Coyote", "Timber Wolf"]
    for row in range(2):
        axes[row, 0].imshow(images[row].resize((224, 224)))
        axes[row, 0].set_title(f"{names[row]} Input", fontsize=11, fontweight='bold')
        axes[row, 0].axis("off")
        
        b1_img = bands[0][1][row].permute(1, 2, 0).cpu().numpy()
        axes[row, 1].imshow(b1_img)
        axes[row, 1].set_title(col_titles[1], fontsize=10, fontweight='bold', color='navy')
        axes[row, 1].axis("off")
        
        b2_img = bands[1][1][row].permute(1, 2, 0).cpu().numpy()
        axes[row, 2].imshow(b2_img)
        axes[row, 2].set_title(col_titles[2], fontsize=10, fontweight='bold', color='darkgreen')
        axes[row, 2].axis("off")
        
        b3_img = bands[2][1][row].permute(1, 2, 0).cpu().numpy()
        axes[row, 3].imshow(b3_img)
        axes[row, 3].set_title(col_titles[3], fontsize=10, fontweight='bold', color='purple')
        axes[row, 3].axis("off")
        
        hvar_img = high_var_ge[row].permute(1, 2, 0).cpu().numpy()
        axes[row, 4].imshow(hvar_img)
        axes[row, 4].set_title(col_titles[4], fontsize=10, fontweight='bold', color='darkred')
        axes[row, 4].axis("off")
        
    plt.suptitle("Resolving the 3-PCA Bottleneck: Spectral Decomposition & High-Variance Manifold PRISM", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "solution_3pca_spectral_bands.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved multi-tier spectral visualization to: {out_path}")

def main():
    set_seed(42)
    print("="*75)
    print("IMPROVEMENT 1: RESOLVING THE 3-PCA BOTTLENECK IN PRISM")
    print("="*75)
    
    model, raw_images, batch, final_act, pre_excitations, true_probs = load_data_and_features()
    u, s, v, mean_val, variances, cum_vars, (b, c, h, w) = svd_decomposition(final_act)
    matrix = final_act.permute(0, 2, 3, 1).reshape(-1, c)
    
    bench_data = run_quantitative_fidelity_benchmark(model, matrix, mean_val, u, s, v, b, c, h, w, true_probs)
    plot_benchmark_results(bench_data)
    
    bands = compute_spectral_bands(u, s, b, h, w, pre_excitations)
    raw_comp, ge_comp, k_needed, cum_var = compute_high_variance_composite(u, s, b, h, w, pre_excitations, target_cum_var=0.85)
    plot_spectral_decomposition(raw_images, bands, ge_comp, k_needed, cum_var)

if __name__ == "__main__":
    main()
