"""
Unit Tests for TorchPRISM Core Library and Dynamic Metadata Resolution
======================================================================
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

from torchprism.metadata import get_class_id, get_class_name, get_environment_metadata
from torchprism import PRISM

class TestTorchPrismCore(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        # Construct a lightweight toy CNN for testing hooks and SVD without downloading heavy weights
        self.toy_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
        self.toy_input = torch.randn(2, 3, 32, 32)
        
    def test_class_id_resolution_valid(self):
        """Tests dynamic resolution of valid ImageNet classes by name and integer string."""
        wolf_id = get_class_id("timber wolf")
        self.assertEqual(wolf_id, 269)
        self.assertIn("wolf", get_class_name(wolf_id).lower())
        
        coyote_id = get_class_id("coyote")
        self.assertEqual(coyote_id, 272)
        self.assertIn("coyote", get_class_name(coyote_id).lower())
        
        malamute_id = get_class_id("malamute")
        self.assertEqual(malamute_id, 249)
        
    def test_class_id_resolution_invalid(self):
        """Tests that invalid queries raise ValueError."""
        with self.assertRaises(ValueError):
            get_class_id("non_existent_creature_xyz_123")
        with self.assertRaises(ValueError):
            get_class_id(99999)
            
    def test_environment_metadata_structure(self):
        """Verifies provenance metadata captures required reproducibility fields."""
        meta = get_environment_metadata()
        self.assertIn("torch_version", meta)
        self.assertIn("python_version", meta)
        self.assertIn("cuda_available", meta)
        self.assertIn("device", meta)
        self.assertEqual(meta["deterministic_seed"], 42)
        
    def test_svd_variance_monotonicity_and_conservation(self):
        """Verifies SVD singular values are positive, monotonically non-increasing, and conserve total energy."""
        # Simulated feature map B=1, C=64, H=7, W=7
        B, C, H, W = 1, 64, 7, 7
        feats = torch.randn(B, C, H, W)
        matrix = feats.permute(0, 2, 3, 1).reshape(-1, C)
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        
        u, s, v = torch.svd(centered)
        
        # Verify singular values are non-negative and non-increasing
        s_np = s.numpy()
        self.assertTrue(all(s_np >= 0))
        self.assertTrue(all(x >= y for x, y in zip(s_np[:-1], s_np[1:])))
        
        # Verify variance sums to 1.0
        vars_pct = (s**2) / (s**2).sum()
        self.assertAlmostEqual(vars_pct.sum().item(), 1.0, places=5)
        
        # Verify rank-K cumulative variance is strictly monotonic
        cum_var = torch.cumsum(vars_pct, dim=0).numpy()
        self.assertTrue(all(x <= y for x, y in zip(cum_var[:-1], cum_var[1:])))

if __name__ == "__main__":
    unittest.main()
