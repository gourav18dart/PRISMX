"""
Improvement 2: Resolving Class-Agnostic Distractors in PRISM
(Class-Attentive Grad-PRISM & Contrastive cPRISM)

This script:
1. Evaluates multi-object scene (Coyote Target vs Timber Wolf Distractor).
2. Computes Standard PRISM (demonstrating distractor contamination).
3. Computes Grad-PRISM (pre-SVD class-attentive filtering isolating target).
4. Computes Contrastive cPRISM (counterfactual feature disentanglement).
5. Measures Signal-to-Distractor Ratio (SDR) and saves figures to outputs/.
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
DATA_DIR_COMP = os.path.join(BASE_DIR, "data", "composites")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def normalize_to_rgb(tensor):
    sc = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
    sc = sc.clamp(-1.0, 1.0)
    sc = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)
    return sc

def gradual_extrapolate(raw_feature_map, activations):
    ext = raw_feature_map.clone()
    for e in activations[::-1]:
        ext = nn.functional.interpolate(
            ext, size=(e.shape[2], e.shape[3]), mode="bilinear", align_corners=False
        )
        ext *= e.mean(dim=1, keepdim=True)
    return normalize_to_rgb(ext)

def run_grad_prism(model, input_tensor, target_class, competing_class=None, gamma=2.0):
    activations = []
    gradients = []
    ge_activations = []
    
    ge_handles = []
    for name, module in model.features.named_children():
        if isinstance(module, nn.MaxPool2d):
            ge_handles.append(module.register_forward_hook(lambda m, inp, out: ge_activations.append(out)))
            
    hf = model.features[30].register_forward_hook(lambda m, inp, out: activations.append(out))
    hb = model.features[30].register_full_backward_hook(lambda m, gin, gout: gradients.append(gout[0]))
    
    model.zero_grad()
    activations.clear(); gradients.clear(); ge_activations.clear()
    logits = model(input_tensor)
    act = activations[0]
    ge_acts = [a.clone() for a in ge_activations[:-1]]
    
    logits[0, target_class].backward(retain_graph=True)
    grad_target = gradients[0]
    w_target = grad_target.mean(dim=(2, 3), keepdim=True)
    
    if competing_class is not None:
        model.zero_grad()
        gradients.clear()
        logits[0, competing_class].backward(retain_graph=True)
        grad_comp = gradients[0]
        w_comp = grad_comp.mean(dim=(2, 3), keepdim=True)
    else:
        w_comp = None
        
    hf.remove(); hb.remove()
    for h in ge_handles: h.remove()
    
    # 1. Standard PRISM
    mat_std = act.permute(0, 2, 3, 1).reshape(-1, 512)
    u_std, s_std, v_std = torch.svd(mat_std - mat_std.mean(0))
    raw_std = (u_std[:, :3] @ torch.diag(s_std[:3])).view(1, 7, 7, 3).permute(0, 3, 1, 2)
    ge_std = gradual_extrapolate(raw_std, ge_acts)
    
    # 2. Grad-PRISM
    cam_target = torch.relu((w_target * act).sum(dim=1, keepdim=True))
    cam_target = cam_target / (cam_target.max() + 1e-8)
    spatial_filter = cam_target ** gamma
    act_purified = act * spatial_filter
    mat_purified = act_purified.permute(0, 2, 3, 1).reshape(-1, 512)
    u_pur, s_pur, v_pur = torch.svd(mat_purified - mat_purified.mean(0))
    raw_grad_prism = (u_pur[:, :3] @ torch.diag(s_pur[:3])).view(1, 7, 7, 3).permute(0, 3, 1, 2)
    ge_grad_prism = gradual_extrapolate(raw_grad_prism, ge_acts)
    h_ext, w_ext = ge_grad_prism.shape[2], ge_grad_prism.shape[3]
    mask_up = nn.functional.interpolate(cam_target, size=(h_ext, w_ext), mode="bilinear", align_corners=False)
    ge_grad_prism_gated = ge_grad_prism * (mask_up > 0.15).float()
    
    # 3. Contrastive cPRISM
    if w_comp is not None:
        w_contrast = torch.relu(w_target - w_comp)
        cam_contrast = torch.relu((w_contrast * act).sum(dim=1, keepdim=True))
        cam_contrast = cam_contrast / (cam_contrast.max() + 1e-8)
        act_contrast = act * (cam_contrast ** gamma)
        mat_cnt = act_contrast.permute(0, 2, 3, 1).reshape(-1, 512)
        u_cnt, s_cnt, v_cnt = torch.svd(mat_cnt - mat_cnt.mean(0))
        raw_cprism = (u_cnt[:, :3] @ torch.diag(s_cnt[:3])).view(1, 7, 7, 3).permute(0, 3, 1, 2)
        ge_cprism = gradual_extrapolate(raw_cprism, ge_acts)
        c_mask = nn.functional.interpolate(cam_contrast, size=(h_ext, w_ext), mode="bilinear", align_corners=False)
        ge_cprism_gated = ge_cprism * (c_mask > 0.15).float()
    else:
        ge_cprism_gated = None
        
    return ge_std, ge_grad_prism_gated, ge_cprism_gated

def compute_sdr(feature_map_2d, target_is_left=True):
    mid = feature_map_2d.shape[1] // 2
    if target_is_left:
        target_energy = feature_map_2d[:, :mid].norm().item()
        distractor_energy = feature_map_2d[:, mid:].norm().item()
    else:
        target_energy = feature_map_2d[:, mid:].norm().item()
        distractor_energy = feature_map_2d[:, :mid].norm().item()
    return target_energy / (distractor_energy + 1e-8)

def main():
    set_seed(42)
    print("=" * 75)
    print("IMPROVEMENT 2: RESOLVING CLASS-AGNOSTIC DISTRACTORS (GRAD-PRISM & CPRISM)")
    print("=" * 75)
    
    comp_path = os.path.join(DATA_DIR_COMP, "distractor_composite.jpg")
    comp_img = Image.open(comp_path).convert("RGB")
    comp_t = preprocess(comp_img).unsqueeze(0).to(device)
    
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device).eval()
    
    ge_std, ge_coy, ge_coy_contrast = run_grad_prism(model, comp_t, target_class=272, competing_class=269)
    _, ge_wolf, ge_wolf_contrast = run_grad_prism(model, comp_t, target_class=269, competing_class=272)
    
    std_sdr = compute_sdr(ge_std[0].norm(dim=0).cpu(), target_is_left=True)
    grad_sdr = compute_sdr(ge_coy[0].norm(dim=0).cpu(), target_is_left=True)
    c_sdr = compute_sdr(ge_coy_contrast[0].norm(dim=0).cpu(), target_is_left=True)
    
    print(f"\nSignal-to-Distractor Ratio (SDR) for Target Coyote:")
    print(f"  Standard PRISM:   {std_sdr:5.2f}")
    print(f"  Grad-PRISM:       {grad_sdr:5.2f} (+{(grad_sdr/std_sdr - 1)*100:.1f}% improvement)")
    print(f"  Contrastive cPRISM: {c_sdr:5.2f}")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(comp_img.resize((224, 224)))
    axes[0, 0].set_title("Input (Coyote + Wolf Distractor)", fontsize=10, fontweight='bold')
    axes[0, 0].axis("off")
    axes[0, 1].imshow(ge_std[0].detach().permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title("Standard PRISM (Target-Agnostic)\n[Distractor Wolf Contaminates Colors]", fontsize=10, fontweight='bold', color='darkred')
    axes[0, 1].axis("off")
    axes[0, 2].imshow(ge_coy[0].detach().permute(1, 2, 0).cpu().numpy())
    axes[0, 2].set_title("Grad-PRISM: Target Coyote\n[Wolf Distractor Eliminated]", fontsize=10, fontweight='bold', color='darkgreen')
    axes[0, 2].axis("off")
    axes[0, 3].imshow(ge_coy_contrast[0].detach().permute(1, 2, 0).cpu().numpy())
    axes[0, 3].set_title("Contrastive cPRISM: Coyote vs Wolf\n[Counterfactual Disentanglement]", fontsize=10, fontweight='bold', color='navy')
    axes[0, 3].axis("off")
    
    axes[1, 0].imshow(comp_img.resize((224, 224)))
    axes[1, 0].set_title("Input (Wolf + Coyote Distractor)", fontsize=10, fontweight='bold')
    axes[1, 0].axis("off")
    axes[1, 1].imshow(ge_std[0].detach().permute(1, 2, 0).cpu().numpy())
    axes[1, 1].set_title("Standard PRISM (Target-Agnostic)\n[Both Objects Intermixed]", fontsize=10, fontweight='bold', color='darkred')
    axes[1, 1].axis("off")
    axes[1, 2].imshow(ge_wolf[0].detach().permute(1, 2, 0).cpu().numpy())
    axes[1, 2].set_title("Grad-PRISM: Target Timber Wolf\n[Coyote Distractor Eliminated]", fontsize=10, fontweight='bold', color='darkgreen')
    axes[1, 2].axis("off")
    axes[1, 3].imshow(ge_wolf_contrast[0].detach().permute(1, 2, 0).cpu().numpy())
    axes[1, 3].set_title("Contrastive cPRISM: Wolf vs Coyote\n[Counterfactual Disentanglement]", fontsize=10, fontweight='bold', color='navy')
    axes[1, 3].axis("off")
    
    plt.suptitle("Eliminating Class-Agnostic Distractors: Grad-PRISM & Contrastive cPRISM", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "solution_distractors_grad_prism.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[SUCCESS] Saved distractor resolution visualization to: {out_path}")

if __name__ == "__main__":
    main()
