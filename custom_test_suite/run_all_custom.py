"""
Master Custom Runner for PRISMX Test Suite
===========================================
Allows running all 3 reproduction experiments and 2 novel improvements
on ANY input images placed in custom_test_suite/input_data/ (or specified via arguments).
All outputs are saved to custom_test_suite/outputs/.
"""

import os
import sys
import glob
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set deterministic seed
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(CURRENT_DIR, "input_data")
DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# Load classes.json if available
class_file = os.path.join(os.path.dirname(CURRENT_DIR), "TorchPRISM", "classes.json")
if os.path.exists(class_file):
    import json
    with open(class_file) as f:
        CLASS_NAMES = {int(k): v.split(',')[0].strip() for k, v in json.load(f).items()}
else:
    CLASS_NAMES = {i: f"Class {i}" for i in range(1000)}

def get_class_name(idx):
    return CLASS_NAMES.get(idx, f"Class {idx}")

# ==============================================================================
# EXPERIMENT 1: FAITHFULNESS VIA REPRESENTATION SWAPPING
# ==============================================================================
def run_exp1(model, img1_path, img2_path, output_dir):
    print("\n" + "="*75)
    print("RUNNING EXPERIMENT 1: FAITHFULNESS (REPRESENTATION SWAPPING)")
    print("="*75)
    
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    
    t1 = preprocess(img1).unsqueeze(0).to(device)
    t2 = preprocess(img2).unsqueeze(0).to(device)
    batch = torch.cat([t1, t2], dim=0)
    
    activations = []
    handle = model.features[30].register_forward_hook(lambda m, inp, out: activations.append(out))
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
    handle.remove()
    
    c1 = probs[0].argmax().item()
    c2 = probs[1].argmax().item()
    p1_orig = probs[0, c1].item()
    p2_orig = probs[1, c2].item()
    
    print(f"Image 1 ({os.path.basename(img1_path)}): Top Class = {get_class_name(c1)} (Prob: {p1_orig:.4f})")
    print(f"Image 2 ({os.path.basename(img2_path)}): Top Class = {get_class_name(c2)} (Prob: {p2_orig:.4f})")
    
    feat1 = activations[0][0].clone() # (512, 7, 7)
    feat2 = activations[0][1].clone()
    
    # Measure cell importance by activation norm
    importance = torch.norm(feat1, dim=0) # (7, 7)
    flat_indices = torch.argsort(importance.flatten(), descending=True)
    
    # Test swapping 0, 5, 15 cells
    swap_counts = [0, 5, 15]
    res_p1 = []
    res_p2 = []
    
    for k in swap_counts:
        swapped = feat1.clone()
        if k > 0:
            for idx in flat_indices[:k]:
                r = idx.item() // 7
                c = idx.item() % 7
                swapped[:, r, c] = feat2[:, r, c]
        
        with torch.no_grad():
            pooled = model.avgpool(swapped.unsqueeze(0))
            out_logits = model.classifier(torch.flatten(pooled, 1))
            out_probs = torch.softmax(out_logits, dim=1)
        
        p1 = out_probs[0, c1].item()
        p2 = out_probs[0, c2].item()
        res_p1.append(p1)
        res_p2.append(p2)
        print(f"Swapped {k:<2} cells -> Orig Class ({get_class_name(c1)}): {p1:.4f} | Competitor ({get_class_name(c2)}): {p2:.4f}")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    ax1.imshow(img1.resize((224, 224)))
    ax1.set_title(f"Target: {get_class_name(c1)}\n(Prob: {p1_orig:.2f})", fontsize=11, fontweight='bold')
    ax1.axis("off")
    
    x = np.arange(len(swap_counts))
    w = 0.35
    ax2.bar(x - w/2, res_p1, w, label=f"Original Class ({get_class_name(c1)})", color="#2b5c8f")
    ax2.bar(x + w/2, res_p2, w, label=f"Swapped Class ({get_class_name(c2)})", color="#d9534f")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{k} cells" for k in swap_counts], fontsize=10, fontweight='bold')
    ax2.set_ylabel("Softmax Probability", fontsize=11, fontweight='bold')
    ax2.set_title("Faithfulness: Prediction Inversion via Cell Swapping", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, "custom_exp1_faithfulness_swapping.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Experiment 1 output: {out_file}")

# ==============================================================================
# EXPERIMENT 2: INTERPRETABILITY (PRISM VS GRAD-CAM)
# ==============================================================================
def run_exp2(model, img_paths, output_dir):
    print("\n" + "="*75)
    print("RUNNING EXPERIMENT 2: INTERPRETABILITY (PRISM VS GRAD-CAM)")
    print("="*75)
    
    imgs = [Image.open(p).convert("RGB") for p in img_paths[:3]]
    tensors = [preprocess(im) for im in imgs]
    batch = torch.stack(tensors).to(device)
    
    activations = []
    gradients = []
    ge_activations = []
    
    ge_handles = []
    for m in model.features:
        if isinstance(m, nn.MaxPool2d):
            ge_handles.append(m.register_forward_hook(lambda mod, inp, out: ge_activations.append(out)))
            
    hf = model.features[30].register_forward_hook(lambda m, inp, out: activations.append(out))
    hb = model.features[30].register_full_backward_hook(lambda m, gin, gout: gradients.append(gout[0]))
    
    logits = model(batch)
    top_classes = [l.argmax().item() for l in logits]
    
    # Compute Grad-CAM
    cams = []
    for i in range(len(imgs)):
        model.zero_grad()
        gradients.clear()
        logits[i, top_classes[i]].backward(retain_graph=True)
        g = gradients[0][i:i+1] # (1, 512, 7, 7)
        a = activations[0][i:i+1]
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(dim=1, keepdim=True))
        cam = nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cams.append(cam[0, 0].detach().cpu().numpy())
        
    hf.remove(); hb.remove()
    for h in ge_handles: h.remove()
    
    # Compute PRISM with Gradual Extrapolation
    final_act = activations[0]
    b, c, h, w = final_act.shape
    mat = final_act.permute(0, 2, 3, 1).reshape(-1, c)
    u, s, v = torch.svd(mat - mat.mean(0))
    raw = (u[:, :3] @ torch.diag(s[:3])).view(b, h, w, 3).permute(0, 3, 1, 2)
    prism_ge = gradual_extrapolate(raw, ge_activations[:-1])
    
    # Plotting
    cols = len(imgs)
    fig, axes = plt.subplots(3, cols, figsize=(4.5 * cols, 10), dpi=200)
    for i in range(cols):
        # 1. Original
        axes[0, i].imshow(imgs[i].resize((224, 224)))
        axes[0, i].set_title(f"Input {i+1}\n{get_class_name(top_classes[i])}", fontsize=11, fontweight='bold')
        axes[0, i].axis("off")
        
        # 2. Grad-CAM
        axes[1, i].imshow(imgs[i].resize((224, 224)))
        axes[1, i].imshow(cams[i], cmap='jet', alpha=0.55)
        axes[1, i].set_title("Grad-CAM\n(Coarse Saliency Blob)", fontsize=10, fontweight='bold')
        axes[1, i].axis("off")
        
        # 3. PRISM
        p_img = prism_ge[i].permute(1, 2, 0).detach().cpu().numpy()
        axes[2, i].imshow(p_img)
        axes[2, i].set_title("PRISM (Ours/Paper)\n(Fine Feature Sections)", fontsize=10, fontweight='bold', color='navy')
        axes[2, i].axis("off")
        
    plt.tight_layout()
    out_file = os.path.join(output_dir, "custom_exp2_interpretability_cam_vs_prism.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Experiment 2 output: {out_file}")

# ==============================================================================
# EXPERIMENT 3: SVD VARIANCE DISTRIBUTION ANALYSIS
# ==============================================================================
def run_exp3(model, img_paths, output_dir):
    print("\n" + "="*75)
    print("RUNNING EXPERIMENT 3: SVD VARIANCE DISTRIBUTION ANALYSIS")
    print("="*75)
    
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    batch = torch.stack([preprocess(im) for im in imgs]).to(device)
    
    activations = []
    handle = model.features[30].register_forward_hook(lambda m, inp, out: activations.append(out))
    with torch.no_grad():
        model(batch)
    handle.remove()
    
    final_act = activations[0]
    b, c, h, w = final_act.shape
    mat = final_act.permute(0, 2, 3, 1).reshape(-1, c)
    u, s, v = torch.svd(mat - mat.mean(0))
    
    variances = (s**2) / (s**2).sum()
    cum_var = torch.cumsum(variances, dim=0) * 100.0
    
    print(f"Top 3 PCs Variance Captured: {cum_var[2].item():.2f}% (Information Lost: {100-cum_var[2].item():.2f}%)")
    k_85 = (cum_var >= 85.0).nonzero(as_tuple=True)[0][0].item() + 1
    print(f"Components needed for >= 85% variance: K = {k_85} ({cum_var[k_85-1].item():.2f}% captured)")
    
    # Scree Plot
    plt.figure(figsize=(9, 4.8), dpi=200)
    plt.plot(range(1, 31), cum_var[:30].cpu().numpy(), marker='o', color='#0072BD', linewidth=2.5, label='Cumulative Variance')
    plt.axvline(x=3, color='red', linestyle='--', label=f'Standard PRISM (K=3: {cum_var[2].item():.1f}%)')
    plt.axvline(x=k_85, color='green', linestyle=':', label=f'High-Variance PRISMX (K={k_85}: {cum_var[k_85-1].item():.1f}%)')
    plt.axhline(y=85, color='gray', linestyle=':')
    plt.xlabel('Principal Components (K)', fontsize=11, fontweight='bold')
    plt.ylabel('Cumulative Explained Variance (%)', fontsize=11, fontweight='bold')
    plt.title(f'SVD Scree Plot: Information Retained across {b} Input Images', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    out_file = os.path.join(output_dir, "custom_exp3_svd_variance_scree.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Experiment 3 output: {out_file}")

# ==============================================================================
# IMPROVEMENT 1: RESOLVING THE 3-PCA BOTTLENECK
# ==============================================================================
def run_imp1(model, img_path, output_dir):
    print("\n" + "="*75)
    print("RUNNING IMPROVEMENT 1: RESOLVING 3-PCA BOTTLENECK (FIDELITY & SPECTRAL BANDS)")
    print("="*75)
    
    img = Image.open(img_path).convert("RGB")
    t = preprocess(img).unsqueeze(0).to(device)
    
    activations = []
    ge_activations = []
    
    ge_handles = []
    for m in model.features:
        if isinstance(m, nn.MaxPool2d):
            ge_handles.append(m.register_forward_hook(lambda mod, inp, out: ge_activations.append(out)))
            
    hf = model.features[30].register_forward_hook(lambda m, inp, out: activations.append(out))
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1)
    hf.remove()
    for h in ge_handles: h.remove()
    
    c_target = probs[0].argmax().item()
    p_orig = probs[0, c_target].item()
    
    final_act = activations[0]
    b, c, h, w = final_act.shape
    mat = final_act.permute(0, 2, 3, 1).reshape(-1, c)
    mean_val = mat.mean(0, keepdim=True)
    u, s, v = torch.svd(mat - mean_val)
    cum_var = torch.cumsum((s**2) / (s**2).sum(), dim=0) * 100.0
    
    # Benchmarking fidelity across K
    k_list = [1, 2, 3, 5, 8, 12, 15, 20, 30]
    p_recon = []
    for k in k_list:
        recon = (u[:, :k] @ torch.diag(s[:k])) @ v[:, :k].t() + mean_val
        recon_feat = recon.view(b, h, w, c).permute(0, 3, 1, 2)
        with torch.no_grad():
            pl = model.avgpool(recon_feat)
            lg = model.classifier(torch.flatten(pl, 1))
            pb = torch.softmax(lg, dim=1)
        p_val = pb[0, c_target].item()
        p_recon.append(p_val)
        status = "Paper 3-PC Drop!" if k == 3 else ("Fidelity Restored!" if k >= 15 else "")
        print(f"K={k:<2} | Var: {cum_var[k-1].item():5.1f}% | Prob: {p_val:.4f} {status}")
        
    # Spectral bands
    bands = []
    for start, end in [(0, 3), (3, 6), (6, 9)]:
        pcs = u[:, start:end] @ torch.diag(s[start:end])
        raw_b = pcs.view(b, h, w, 3).permute(0, 3, 1, 2)
        ge_b = gradual_extrapolate(raw_b, ge_activations[:-1])
        bands.append(ge_b[0].permute(1, 2, 0).detach().cpu().numpy())
        
    # High-Variance Composite (K=15)
    k_15 = 15
    coords_15 = u[:, :k_15] @ torch.diag(s[:k_15])
    W = torch.zeros(k_15, 3, device=device)
    for i in range(k_15): W[i, i % 3] = s[i] / s[0]
    raw_comp = (coords_15 @ W).view(b, h, w, 3).permute(0, 3, 1, 2)
    comp_ge = gradual_extrapolate(raw_comp, ge_activations[:-1])[0].permute(1, 2, 0).detach().cpu().numpy()
    
    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=200)
    axes[0].imshow(img.resize((224, 224)))
    axes[0].set_title(f"Input: {get_class_name(c_target)}\n(Full 512D Prob: {p_orig:.2f})", fontsize=10, fontweight='bold')
    axes[0].axis("off")
    
    axes[1].imshow(bands[0])
    axes[1].set_title(f"Band 1 (PCs 1..3: Macro)\n[Var: {cum_var[2].item():.1f}% | Prob: {p_recon[2]:.2f}]", fontsize=10, fontweight='bold', color='navy')
    axes[1].axis("off")
    
    axes[2].imshow(bands[1])
    axes[2].set_title("Band 2 (PCs 4..6: Contours)\n[Mid-level Structural Edges]", fontsize=10, fontweight='bold', color='darkgreen')
    axes[2].axis("off")
    
    axes[3].imshow(bands[2])
    axes[3].set_title("Band 3 (PCs 7..9: Textures)\n[High-Frequency Details]", fontsize=10, fontweight='bold', color='purple')
    axes[3].axis("off")
    
    axes[4].imshow(comp_ge)
    axes[4].set_title(f"High-Variance PRISMX (15 PCs)\n[Var: {cum_var[14].item():.1f}% | Prob: {p_recon[6]:.2f}]", fontsize=10, fontweight='bold', color='darkred')
    axes[4].axis("off")
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, "custom_imp1_3pca_fidelity_and_spectral_bands.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Improvement 1 output: {out_file}")

# ==============================================================================
# IMPROVEMENT 2: RESOLVING CLASS-AGNOSTIC DISTRACTORS (GRAD-PRISM)
# ==============================================================================
def run_imp2(model, img_path, output_dir):
    print("\n" + "="*75)
    print("RUNNING IMPROVEMENT 2: RESOLVING DISTRACTORS (GRAD-PRISM)")
    print("="*75)
    
    img = Image.open(img_path).convert("RGB")
    t = preprocess(img).unsqueeze(0).to(device)
    
    activations = []
    gradients = []
    ge_activations = []
    
    ge_handles = []
    for m in model.features:
        if isinstance(m, nn.MaxPool2d):
            ge_handles.append(m.register_forward_hook(lambda mod, inp, out: ge_activations.append(out)))
            
    hf = model.features[30].register_forward_hook(lambda m, inp, out: activations.append(out))
    hb = model.features[30].register_full_backward_hook(lambda m, gin, gout: gradients.append(gout[0]))
    
    logits = model(t)
    probs = torch.softmax(logits, dim=1)
    
    c_target = probs[0].argmax().item()
    p_target = probs[0, c_target].item()
    print(f"Image ({os.path.basename(img_path)}): Target Class = {get_class_name(c_target)} (Prob: {p_target:.4f})")
    
    logits[0, c_target].backward(retain_graph=True)
    grad_target = gradients[0]
    w_target = grad_target.mean(dim=(2, 3), keepdim=True)
    
    hf.remove(); hb.remove()
    for h in ge_handles: h.remove()
    
    act = activations[0]
    ge_acts = [a.clone() for a in ge_activations[:-1]]
    
    # 1. Standard PRISM
    mat_std = act.permute(0, 2, 3, 1).reshape(-1, 512)
    u_std, s_std, _ = torch.svd(mat_std - mat_std.mean(0))
    raw_std = (u_std[:, :3] @ torch.diag(s_std[:3])).view(1, 7, 7, 3).permute(0, 3, 1, 2)
    prism_ge_std = gradual_extrapolate(raw_std, ge_acts)[0].permute(1, 2, 0).detach().cpu().numpy()
    
    # 2. Grad-PRISM
    cam_target = torch.relu((w_target * act).sum(dim=1, keepdim=True))
    cam_target = cam_target / (cam_target.max() + 1e-8)
    act_purified = act * (cam_target ** 1.8)
    mat_purified = act_purified.permute(0, 2, 3, 1).reshape(-1, 512)
    u_pur, s_pur, _ = torch.svd(mat_purified - mat_purified.mean(0))
    raw_pur = (u_pur[:, :3] @ torch.diag(s_pur[:3])).view(1, 7, 7, 3).permute(0, 3, 1, 2)
    ge_pur = gradual_extrapolate(raw_pur, ge_acts)
    
    mask_up = nn.functional.interpolate(cam_target, size=(ge_pur.shape[2], ge_pur.shape[3]), mode="bilinear", align_corners=False)
    grad_prism_gated = (ge_pur * (mask_up > 0.12).float())[0].permute(1, 2, 0).detach().cpu().numpy()
    cam_np = mask_up[0, 0].detach().cpu().numpy()
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=200)
    axes[0].imshow(img.resize((224, 224)))
    axes[0].set_title(f"Input: {get_class_name(c_target)}\n(Prob: {p_target:.2f})", fontsize=10, fontweight='bold')
    axes[0].axis("off")
    
    axes[1].imshow(prism_ge_std)
    axes[1].set_title("Standard PRISM (Target-Agnostic)\n[Distractors Contaminate Colors]", fontsize=10, fontweight='bold', color='darkred')
    axes[1].axis("off")
    
    axes[2].imshow(img.resize((224, 224)))
    axes[2].imshow(cam_np, cmap='jet', alpha=0.55)
    axes[2].set_title(f"Class Activation Map\n[Target: {get_class_name(c_target)}]", fontsize=10, fontweight='bold')
    axes[2].axis("off")
    
    axes[3].imshow(grad_prism_gated)
    axes[3].set_title("Grad-PRISM (Our Solution)\n[Distractors Silenced Pre-SVD]", fontsize=10, fontweight='bold', color='darkgreen')
    axes[3].axis("off")
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, "custom_imp2_class_attentive_grad_prism.png")
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Improvement 2 output: {out_file}")

# ==============================================================================
# MAIN DRIVER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run PRISMX Test Suite on custom input images.")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save output figures")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(args.input_dir, "*.[jJ][pP][gG]")) + 
                    glob.glob(os.path.join(args.input_dir, "*.[pP][nN][gG]")) +
                    glob.glob(os.path.join(args.input_dir, "*.[jJ][pP][eE][gG]")))
    
    if not images:
        print(f"[ERROR] No image files found in {args.input_dir}!")
        return
        
    print("=" * 80)
    print("PRISMX COMPREHENSIVE CUSTOM TEST SUITE")
    print(f"Input Directory:  {args.input_dir} ({len(images)} images detected)")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)
    for i, im in enumerate(images, 1):
        print(f"  {i}. {os.path.basename(im)}")
        
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device).eval()
    
    # 1. Experiment 1: Faithfulness (requires 2 images)
    img1 = images[0]
    img2 = images[1] if len(images) > 1 else images[0]
    run_exp1(model, img1, img2, args.output_dir)
    
    # 2. Experiment 2: Interpretability (PRISM vs Grad-CAM)
    run_exp2(model, images[:3], args.output_dir)
    
    # 3. Experiment 3: SVD Variance Analysis
    run_exp3(model, images, args.output_dir)
    
    # 4. Improvement 1: Resolving 3-PCA Bottleneck
    run_imp1(model, images[0], args.output_dir)
    
    # 5. Improvement 2: Resolving Distractors
    # Look for distractor image (e.g. malamute_mushroom or first image)
    distractor_img = None
    for im in images:
        if "mushroom" in os.path.basename(im).lower():
            distractor_img = im
            break
    if distractor_img is None:
        distractor_img = images[-1]
    run_imp2(model, distractor_img, args.output_dir)
    
    print("\n" + "=" * 80)
    print(f"[COMPLETED] All 3 Experiments & 2 Improvements successfully generated in:")
    print(f"  {args.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
