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
    # Create a dummy ExifTags module if not available
    import sys
    from types import ModuleType

    ExifTags = ModuleType('ExifTags')
    ExifTags.TAGS = {}
    sys.modules['PIL.ExifTags'] = ExifTags


class MMBench(Dataset):
    def __init__(self, image_preprocess=None, groups_filter: Optional[List[str]] = None,
                 n_samples: int = 100, seed: int = 42, root_dir='data', download=False):
        """
        MMBench Dataset.

        Args:
            image_preprocess: Image preprocessing function
            groups_filter: List of categories to include (None for all)
            n_samples: Number of samples per category
            seed: Random seed for sampling
            root_dir: Root directory for data
            download: Whether to download dataset if not found
        """
        self.image_preprocess = image_preprocess
        self.root_dir = root_dir

        # Load the dataset
        self.data = self._load_mmbench(groups_filter, n_samples, seed)

        print(f"Loaded MMBench: {len(self.data)} samples")

    def _load_mmbench(self, groups_filter: Optional[List[str]], n_samples: int, seed: int = 42) -> List[Dict]:
        from datasets import load_dataset

        print("Loading MMBench (lmms-lab/MMBench en dev)…")
        ds = load_dataset("lmms-lab/MMBench", "en", split="dev")

        by_group: Dict[str, List[Dict]] = defaultdict(list)
        for row in ds:
            cat = str(row.get("category", "unknown")).strip()
            if groups_filter and cat not in groups_filter:
                continue
            opts = {l: str(row.get(l, "") or "").strip()
                    for l in ["A", "B", "C", "D"]
                    if str(row.get(l, "") or "").strip() and str(row.get(l, "") or "").strip().lower() != "nan"}
            if not opts:
                continue
            gt = str(row.get("answer", "")).strip().upper()
            question = str(row.get("question", "")).strip()
            hint = str(row.get("hint", "") or "").strip()
            if hint and hint.lower() != "nan":
                question = f"{hint}\n{question}"
            opt_text = "\n".join(f"{k}. {v}" for k, v in opts.items())
            prompt = f"{question}\n{opt_text}\nAnswer with the option's letter from the given choices directly."
            by_group[cat].append({
                "image": row["image"].convert("RGB"),
                "prompt": prompt,
                "ground_truth": gt,
                "group": cat,
            })

        rng = random.Random(seed)
        out = []
        for cat, items in sorted(by_group.items()):
            selected = rng.sample(items, min(n_samples, len(items)))
            out.extend(selected)
        print(f"  {len(out)} samples, groups: {sorted(by_group.keys())}")
        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        image = item["image"]
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Return in the format expected by the model
        # For MMBench, we have a single image and a single prompt
        return edict({
            "image_options": [image],
            "caption_options": [item["prompt"]],
            "ground_truth": item["ground_truth"],
            "group": item["group"]
        })

    def save_scores(self, scores, correct_id, path, dataset, method, weight, model_name, option):
        """
        Save scores to JSON file.

        Args:
            scores: Accuracy or scores
            correct_id: List of correct indices
            path: Output directory
            dataset: Dataset name
            method: Method name
            weight: Weight used
            model_name: Model name
            option: Option used
        """
        import json
        path_ = os.path.join(path, 'res.json')
        data = {
            "dataset": dataset,
            "model": model_name,
            "option": option,
            "method": method,
            "weight": weight,
            "Individual accuracy": scores,
            "correct_id": correct_id
        }
        with open(path_, 'a+') as file:
            json.dump(data, file)
            file.write('\n')

    def evaluate_scores(self, scores, path, dataset, model, method, weight, sampled_indices, option):
        """
        Evaluate scores for MMBench.

        Args:
            scores: Model scores
            path: Output directory
            dataset: Dataset name
            model: Model name
            method: Method name
            weight: Weight used
            sampled_indices: Indices of sampled data
            option: Option used

        Returns:
            List of result records
        """
        # For MMBench, we need to extract the answer from the model output
        # This would typically involve parsing the model's text response
        # For now, we'll save the scores and let the evaluation be done separately

        import json
        path_ = os.path.join(path, 'res.json')

        # Calculate accuracy if scores are predictions
        if hasattr(self, 'predictions'):
            correct = sum(1 for i, pred in enumerate(self.predictions)
                         if pred == self.data[i].get('ground_truth'))
            accuracy = correct / len(self.predictions) if self.predictions else 0

            data = {
                "dataset": dataset,
                "model": model,
                "option": option,
                "method": method,
                "weight": weight,
                "accuracy": accuracy
            }
        else:
            # Save raw scores
            data = {
                "dataset": dataset,
                "model": model,
                "option": option,
                "method": method,
                "weight": weight,
                "scores": scores
            }

        with open(path_, 'a+') as file:
            json.dump(data, file)
            file.write('\n')

        return data


def get_mmbench(image_preprocess=None, groups_filter=None, n_samples=100, seed=42, download=False):
    """
    Get MMBench dataset.

    Args:
        image_preprocess: Image preprocessing function
        groups_filter: List of categories to include (None for all)
        n_samples: Number of samples per category
        seed: Random seed for sampling
        download: Whether to download dataset if not found (for compatibility)

    Returns:
        MMBench dataset instance
    """
    return MMBench(
        image_preprocess=image_preprocess,
        groups_filter=groups_filter,
        n_samples=n_samples,
        seed=seed
    )
