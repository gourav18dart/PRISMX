"""
PRISMX Metadata & Experiment Provenance System
==============================================
Provides dynamic ImageNet class resolution (eliminating hardcoded magic numbers)
and automated tracking of model versions, weights, hardware, and experiment metrics.
"""

import os
import json
import time
import platform
import torch
import torchvision

_METADATA_DIR = os.path.dirname(os.path.abspath(__file__))
_CLASSES_PATH = os.path.join(_METADATA_DIR, "classes.json")
_CLASSES_MAP = None

def _load_classes():
    global _CLASSES_MAP
    if _CLASSES_MAP is None:
        if os.path.exists(_CLASSES_PATH):
            with open(_CLASSES_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
                _CLASSES_MAP = {int(k): v for k, v in raw.items()}
        else:
            _CLASSES_MAP = {}
    return _CLASSES_MAP

def get_class_id(query):
    """
    Dynamically resolves an ImageNet class index from an integer ID or class name query.
    
    Args:
        query (int or str): Class ID (int or int-string) or substring search (e.g., "timber wolf", "coyote").
        
    Returns:
        int: ImageNet class index (0-999).
        
    Raises:
        ValueError: If query cannot be resolved to a valid class index.
    """
    classes = _load_classes()
    
    if isinstance(query, int):
        if 0 <= query < len(classes):
            return query
        raise ValueError(f"Class index {query} out of range [0, {len(classes)-1}]")
        
    if isinstance(query, str) and query.isdigit():
        idx = int(query)
        if 0 <= idx < len(classes):
            return idx
        raise ValueError(f"Class index {idx} out of range [0, {len(classes)-1}]")
        
    query_norm = str(query).lower().strip()
    
    # 1. Exact match against comma-separated tokens
    for idx, name in classes.items():
        tokens = [t.strip().lower() for t in name.split(",")]
        if query_norm in tokens:
            return idx
            
    # 2. Substring match
    for idx, name in classes.items():
        if query_norm in name.lower():
            return idx
            
    raise ValueError(f"Could not resolve ImageNet class for query: '{query}'")

def get_class_name(class_id):
    """
    Returns the canonical descriptive name for an ImageNet class index.
    """
    classes = _load_classes()
    idx = int(class_id)
    return classes.get(idx, f"Unknown class {idx}")

def get_environment_metadata():
    """
    Gathers reproducible runtime environment and hardware details.
    """
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": cuda_available,
        "device": device_name,
        "default_model": "VGG16",
        "default_weights": "VGG16_Weights.IMAGENET1K_V1",
        "deterministic_seed": 42
    }

def record_experiment_metadata(script_name, metrics=None, config=None, output_path="outputs/experiment_metadata.json"):
    """
    Appends an experiment run entry into outputs/experiment_metadata.json for provenance.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    records = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                records = json.load(f)
                if not isinstance(records, list):
                    records = [records]
        except Exception:
            records = []
            
    entry = {
        "script": script_name,
        "environment": get_environment_metadata(),
        "config": config or {},
        "metrics": metrics or {}
    }
    
    # Update existing entry for the script or append new
    updated = False
    for i, r in enumerate(records):
        if r.get("script") == script_name:
            records[i] = entry
            updated = True
            break
    if not updated:
        records.append(entry)
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        
    return entry
