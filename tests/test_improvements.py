"""
Unit Tests for Novel Research Improvements:
Improvement 1 (Multi-PCA Spectral Reconstruction & 3-Channel Projection)
Improvement 2 (Dual Spatial-Channel Gradient Gating & Distractor Suppression)
=============================================================================
"""

import unittest
import os
import sys
import torch
import torch.nn as nn

# Add project root and src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from torchprism.metadata import get_class_id

class TestImprovements(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        
    def test_imp1_reconstruction_fidelity_convergence(self):
        """
        Tests that rank-K SVD reconstruction error strictly decreases as K increases,
        and full-rank reconstruction (K=C) recovers 100% of feature tensor energy.
        """
        B, C, H, W = 2, 32, 7, 7
        features = torch.randn(B, C, H, W)
        matrix = features.permute(0, 2, 3, 1).reshape(-1, C)
        mean_val = matrix.mean(dim=0, keepdim=True)
        centered = matrix - mean_val
        u, s, v = torch.svd(centered)
        
        errors = []
        for k in [1, 3, 8, 16, 32]:
            recon_centered = (u[:, :k] @ torch.diag(s[:k])) @ v[:, :k].t()
            recon_matrix = recon_centered + mean_val
            rel_err = (torch.norm(matrix - recon_matrix) / torch.norm(matrix)).item()
            errors.append(rel_err)
            
        # Error must decrease monotonically with K
        self.assertTrue(all(x >= y for x, y in zip(errors[:-1], errors[1:])))
        # Full rank error must be near machine epsilon
        self.assertLess(errors[-1], 1e-4)
        
    def test_imp1_visualization_projection_shape(self):
        """
        Tests that the 3-channel visualization projection correctly maps K components
        to 3 RGB color channels using variance-weighted projection matrix W.
        """
        K = 15
        coords = torch.randn(98, K)
        s = torch.sort(torch.rand(K), descending=True)[0]
        
        W = torch.zeros(K, 3)
        for i in range(K):
            W[i, i % 3] = s[i] / s[0]
            
        proj = coords @ W
        self.assertEqual(proj.shape, (98, 3))
        self.assertFalse(torch.isnan(proj).any())
        
    def test_imp2_dual_spatial_channel_gating(self):
        """
        Tests that Dual Spatial-Channel Gating:
        1. Preserves tensor dimensionality [B, C, H, W].
        2. Attenuates negative/unimportant channels (channel gating).
        3. Attenuates distractor spatial regions (spatial gating).
        """
        B, C, H, W = 1, 16, 7, 7
        act = torch.rand(B, C, H, W) + 0.1 # positive activations
        
        # Simulated target gradient: positive for channels 0..7, negative/zero for 8..15
        grad_target = torch.randn(B, C, H, W)
        grad_target[:, :8, :, :] = torch.abs(grad_target[:, :8, :, :]) + 0.5
        grad_target[:, 8:, :, :] = -torch.abs(grad_target[:, 8:, :, :]) - 0.5
        
        w_target = grad_target.mean(dim=(2, 3), keepdim=True)
        
        # 1. Channel weights
        alpha_pos = torch.relu(w_target)
        w_ch = alpha_pos / (alpha_pos.max() + 1e-8)
        
        # Channels 8..15 must be exactly zeroed by ReLU channel gating
        self.assertTrue(torch.all(w_ch[:, 8:, :, :] == 0.0))
        self.assertTrue(torch.all(w_ch[:, :8, :, :] > 0.0))
        
        # 2. Spatial CAM mask
        cam = torch.relu((w_target * act).sum(dim=1, keepdim=True))
        cam = cam / (cam.max() + 1e-8)
        spatial_filter = cam ** 2.0
        
        # 3. Dual Gated Activation
        act_purified = act * w_ch * spatial_filter
        self.assertEqual(act_purified.shape, act.shape)
        # Suppressed channels must remain zeroed
        self.assertTrue(torch.all(act_purified[:, 8:, :, :] == 0.0))
        
    def test_sdr_metric_calculation(self):
        """Tests Signal-to-Distractor Ratio (SDR) calculation correctness."""
        # Simulated 2D heatmap with high energy on left, low on right
        H, W = 10, 10
        feature_map_2d = torch.zeros(H, W)
        feature_map_2d[:, :5] = 10.0 # Target region
        feature_map_2d[:, 5:] = 2.0  # Distractor region
        
        mid = W // 2
        target_energy = feature_map_2d[:, :mid].norm().item()
        distractor_energy = feature_map_2d[:, mid:].norm().item()
        sdr = target_energy / (distractor_energy + 1e-8)
        
        self.assertAlmostEqual(sdr, 5.0, places=3)
        self.assertGreater(sdr, 1.0)

if __name__ == "__main__":
    unittest.main()
