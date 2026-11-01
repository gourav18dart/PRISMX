"""
Master Test Runner for PRISMX Test Suite
========================================
Runs all unit tests in tests/ directory and reports results.
"""

import unittest
import sys
import os

if __name__ == "__main__":
    test_dir = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.defaultTestLoader.discover(start_dir=test_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
