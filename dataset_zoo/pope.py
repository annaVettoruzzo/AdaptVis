import os
import json
import random
from typing import List, Dict, Optional
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from easydict import EasyDict as edict

# Fix PIL ExifTags issue for HuggingFace datasets
try:
    from PIL import ExifTags
except ImportError:
    import sys
    from types import ModuleType
    ExifTags = ModuleType('ExifTags')
    ExifTags.TAGS = {}
    sys.modules['PIL.ExifTags'] = ExifTags


class POPE(Dataset):
    def __init__(self, image_preprocess=None, groups_filter: Optional[List[str]] = None,
                 n_samples: int = 100, seed: int = 42, root_dir='data', download=False):
        """
        POPE Dataset.

        Args:
            image_preprocess: Image preprocessing function
            groups_filter: List of categories to include (adversarial, popular, random)
            n_samples: Number of samples per category
            seed: Random seed for sampling
            root_dir: Root directory for data
            download: Whether to download dataset if not found
        """
        self.image_preprocess = image_preprocess
        self.root_dir = root_dir

        self.data = self._load_pope(groups_filter, n_samples, seed)

        print(f"Loaded POPE: {len(self.data)} samples")

    def _load_pope(self, groups_filter: Optional[List[str]], n_samples: int, seed: int = 42) -> List[Dict]:
        from datasets import load_dataset

        print("Loading POPE (lmms-lab/POPE)...")
        ds = load_dataset("lmms-lab/POPE", split="test")

        by_group: Dict[str, List[Dict]] = defaultdict(list)
        for row in ds:
            cat = str(row.get("category", "unknown")).strip()
            if groups_filter and cat not in groups_filter:
                continue
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip().lower()
            by_group[cat].append({
                "image": row["image"].convert("RGB"),
                "question": question,
                "answer": answer,
                "category": cat,
            })

        rng = random.Random(seed)
        out = []
        for cat, items in sorted(by_group.items()):
            selected = rng.sample(items, min(n_samples, len(items)))
            out.extend(selected)
        print(f"  {len(out)} samples, categories: {sorted(by_group.keys())}")
        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        image = item["image"]
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        return edict({
            "image_options": [image],
            "caption_options": [item["question"]],
            "ground_truth": item["answer"],
            "group": item["category"]
        })


def get_pope(image_preprocess=None, groups_filter=None, n_samples=100, seed=42, download=False):
    """
    Get POPE dataset.

    Args:
        image_preprocess: Image preprocessing function
        groups_filter: List of categories to include (None for all)
        n_samples: Number of samples per category
        seed: Random seed for sampling
        download: Whether to download dataset if not found (for compatibility)

    Returns:
        POPE dataset instance
    """
    return POPE(
        image_preprocess=image_preprocess,
        groups_filter=groups_filter,
        n_samples=n_samples,
        seed=seed
    )
