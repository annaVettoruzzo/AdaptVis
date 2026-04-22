"""
Batch visualize attention rollout for all MMBench samples.

Usage:
    python visualize_all_mmbench.py
    python visualize_all_mmbench.py --base-dir ./output/mmbench_base
    python visualize_all_mmbench.py --max-samples 10
"""

import os
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Batch visualize attention rollout')
    parser.add_argument("--base-dir", default="./output/mmbench_base",
            help="Directory containing attention maps")
    parser.add_argument("--output-dir", default="./rollout_outputs/mmbench",
            help="Output directory for visualizations")
    parser.add_argument("--max-samples", default=None, type=int,
            help="Maximum number of samples to process (default: all)")
    parser.add_argument("--start", default=0, type=int,
            help="Start from this sample index (default: 0)")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # Check if base directory exists
    if not base_dir.exists():
        print(f"❌ Error: {base_dir} not found. Run rollout_mmbench.py first.")
        return 1

    # Find all sample directories
    sample_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and (d.name).isdigit()])

    if not sample_dirs:
        print(f"❌ Error: No samples found in {base_dir}")
        return 1

    # Filter by range if specified
    if args.max_samples:
        sample_dirs = sample_dirs[args.start:args.start + args.max_samples]
    else:
        sample_dirs = sample_dirs[args.start:]

    total = len(sample_dirs)

    print(f"Found {total} samples to process")
    print("=" * 60)

    # Process each sample
    success_count = 0
    fail_count = 0

    for idx, sample_dir in enumerate(sample_dirs):
        print(f"\n[{idx+1}/{total}] Processing sample {sample_dir.name}...")

        try:
            cmd = [
                "python", "compute_attention_rollout.py",
                "--save_dir", str(sample_dir),
                "--output_dir", args.output_dir,
                "--show_evolution"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ Sample {sample_dir.name} completed")
                success_count += 1
            else:
                print(f"✗ Sample {sample_dir.name} failed")
                print(f"  Error: {result.stderr[:200]}")
                fail_count += 1

        except Exception as e:
            print(f"✗ Sample {sample_dir.name} failed: {e}")
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"✓ Processing completed!")
    print(f"  Successful: {success_count}/{total}")
    print(f"  Failed: {fail_count}/{total}")
    print(f"\nOutputs saved to: {args.output_dir}")
    print("\nView results:")
    print(f"  - Individual heatmaps: {args.output_dir}/{{idx}}/attention_rollout.png")
    print(f"  - Layer evolution: {args.output_dir}/{{idx}}/attention_evolution.png")
    print("=" * 60)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
