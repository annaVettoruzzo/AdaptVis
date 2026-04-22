"""
Inspect MMBench dataset to show all available categories/groups and their counts.

Usage:
    python inspect_mmbench_categories.py
"""

from datasets import load_dataset
from collections import Counter

# Fix for PIL ExifTags issue (same as in mmbench.py)
try:
    from PIL import Image
    if not hasattr(Image, 'ExifTags'):
        import sys
        from types import ModuleType
        ExifTags = ModuleType('ExifTags')
        ExifTags.TAGS = {}
        ExifTags.Base = type('Base', (), {'Orientation': 274})  # Common EXIF tag
        sys.modules['PIL.ExifTags'] = ExifTags
        Image.ExifTags = ExifTags
except Exception:
    pass  # Ignore any PIL-related errors

def main():
    print("=" * 60)
    print("MMBench Dataset - Categories/Groups Inspection")
    print("=" * 60)

    print("\nLoading MMBench dataset from HuggingFace...")
    try:
        dataset = load_dataset("lmms-lab/MMBench", "en", split="dev")
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    # Check for category/group field
    print("\nChecking dataset fields...")
    print(f"Available columns: {dataset.column_names}")

    # Look for common category field names
    category_field = None
    for field in ['category', 'group', 'categories', 'groups', 'l2-category']:
        if field in dataset.column_names:
            category_field = field
            break

    if not category_field:
        print("\n⚠ No standard category field found. Trying to detect from data...")
        # Print a sample to see what fields are available
        print("\nSample data:")
        sample = dataset[0]
        for key, value in sample.items():
            if isinstance(value, str):
                print(f"  {key}: {value[:100] if len(value) > 100 else value}")
            elif isinstance(value, list) and len(value) > 0:
                print(f"  {key}: [list with {len(value)} items]")
            else:
                print(f"  {key}: {value}")

        # Try to find categorical data
        for key in dataset.column_names:
            unique_values = set(dataset[:100][key])  # Check first 100
            if len(unique_values) < 50 and len(unique_values) > 1:  # Likely categorical
                print(f"\n  → Possible category field: '{key}'")
                print(f"     Unique values: {list(unique_values)[:10]}")

                # Ask if this looks like categories
                response = input(f"\nUse '{key}' as category field? (y/n): ")
                if response.lower() == 'y':
                    category_field = key
                    break

    if category_field:
        print(f"\n✓ Using category field: '{category_field}'")

        # Get all categories and their counts
        categories = [item[category_field] for item in dataset]
        category_counts = Counter(categories)

        print(f"\nFound {len(category_counts)} unique categories:")
        print("-" * 60)

        # Sort by count (descending)
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

        for category, count in sorted_categories:
            print(f"  {category:30s} : {count:4d} samples")

        print("-" * 60)
        print(f"  {'TOTAL':30s} : {len(dataset):4d} samples")

        print("\n" + "=" * 60)
        print("Usage with --groups-filter:")
        print("  python rollout_mmbench.py --groups-filter \"Reasoning,Math\"")
        print("\nAvailable categories to use:")
        print(f"  {', '.join([cat for cat, _ in sorted_categories])}")
        print("=" * 60)

    else:
        print("\n✗ Could not determine category field automatically.")
        print("\nYou may need to inspect the dataset manually:")
        print("  from datasets import load_dataset")
        print("  ds = load_dataset('lmms-lab/MMBench', 'en', split='dev')")
        print("  print(ds[0].keys())")


if __name__ == "__main__":
    main()
