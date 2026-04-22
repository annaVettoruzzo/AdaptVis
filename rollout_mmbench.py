"""
Compute attention rollout for MMBench dataset using base LLaVA 1.5 model.

This script:
1. Loads MMBench dataset
2. Runs LLaVA 1.5 base model (weight=1.0 = no manipulation)
3. Saves attention maps for each sample
4. You can then use compute_attention_rollout.py to visualize

Usage:
    # Run on 10 samples per category (for testing)
    python rollout_mmbench.py --n-samples 10

    # Run on specific categories
    python rollout_mmbench.py --groups-filter "Reasoning,Math" --n-samples 50

    # Run on all categories with default 100 samples each
    python rollout_mmbench.py
"""

import os
import sys
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

def main():
    parser = argparse.ArgumentParser(description='Compute attention rollout for MMBench')
    parser.add_argument("--groups-filter", default=None, type=str,
            help="Comma-separated list of categories to include (e.g., 'Reasoning,Math')")
    parser.add_argument("--n-samples", default=10, type=int,
            help="Number of samples per category (default: 10 for testing)")
    parser.add_argument("--seed", default=42, type=int,
            help="Random seed")
    parser.add_argument("--output-dir", default="./output/mmbench_base",
            help="Output directory for attention maps")
    parser.add_argument("--batch-size", default=1, type=int,
            help="Batch size (keep at 1 for attention saving)")
    parser.add_argument("--device", default="cuda", type=str,
            help="Device to use")

    args = parser.parse_args()

    # Import after argparse to avoid slow imports
    from model_zoo import get_model
    from dataset_zoo import get_dataset

    print("=" * 60)
    print("MMBench Attention Rollout - Base LLaVA 1.5")
    print("=" * 60)

    # Parse MMBench arguments
    dataset_kwargs = {}
    if args.groups_filter:
        dataset_kwargs["groups_filter"] = [g.strip() for g in args.groups_filter.split(',')]
    else:
        dataset_kwargs["groups_filter"] = None
    dataset_kwargs["n_samples"] = args.n_samples
    dataset_kwargs["seed"] = args.seed

    # Load model (use scaling_vis with weight=1.0 for base model + attention saving)
    print("\n1. Loading LLaVA 1.5 model (base, with attention saving)...")
    model, image_preprocess = get_model("llava1.5", args.device, method="scaling_vis")
    print("   ✓ Model loaded")

    # Initialize the attention saving mechanism
    from model_zoo.llava15 import change_greedy_to_add_weight
    change_greedy_to_add_weight()
    print("   ✓ Attention saving initialized")

    # Load MMBench dataset
    print(f"\n2. Loading MMBench dataset...")
    print(f"   Categories: {args.groups_filter if args.groups_filter else 'All'}")
    print(f"   Samples per category: {args.n_samples}")

    # Note: image_preprocess is None for LLaVA, so we need to handle PIL Images in collate
    from misc import _default_collate
    dataset = get_dataset("MMBench", image_preprocess=image_preprocess, **dataset_kwargs)
    print(f"   ✓ Dataset loaded: {len(dataset)} samples")

    # Create dataloader with custom collate function to handle PIL Images
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_default_collate)

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Process each sample
    print(f"\n3. Processing samples and saving attention maps...")
    print(f"   Output directory: {output_dir}")

    results = []

    for idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
        # Set save path for this sample
        sample_dir = f"{output_dir}/{idx}/"
        os.environ['SAVE_ATTN_PATH'] = sample_dir
        os.makedirs(sample_dir, exist_ok=True)

        # Get image and prompt
        image = batch["image_options"][0][0]  # First image option, first (and only) image
        mmbench_prompt = batch["caption_options"][0][0]  # First (and only) caption

        # Save original image for visualization overlay
        image_path = f"{sample_dir}/original_image.png"
        image.save(image_path)

        # Format prompt for LLaVA 1.5 (needs conversation format with <image> token)
        prompt = f"USER: <image>\n{mmbench_prompt}\nASSISTANT:"

        # Prepare input (LLaVA processor handles the image token)
        inputs = model.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(model.device)

        # Create key mask for special token (image token)
        keys = [torch.where(input_id == 32001, 1, 0) for input_id in inputs['input_ids']]

        # Generate with weight=1.0 (equivalent to base model)
        # Using scaling_vis with weight=1.0 enables attention saving without manipulation
        with torch.no_grad():
            output = model.model.generate(
                **inputs,
                keys=keys,
                weight=1.0,  # weight=1.0 = no attention manipulation (base model)
                max_new_tokens=100,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Decode generation
        generated_text = model.processor.decode(
            output['sequences'][0][len(inputs['input_ids'][-1]):],
            skip_special_tokens=True
        )

        # Save result
        result = {
            "index": idx,
            "prompt": prompt,
            "generation": generated_text,
        }

        # Add ground truth if available
        if 'ground_truth' in batch:
            result["ground_truth"] = batch["ground_truth"][0]
        if 'group' in batch:
            result["category"] = batch["group"][0]

        results.append(result)

        # Print progress every 10 samples
        if (idx + 1) % 10 == 0:
            print(f"\n   Processed {idx + 1}/{len(dataset)} samples")
            print(f"   Latest generation: {generated_text[:100]}...")

    # Save all results
    results_file = f"{output_dir}/results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n" + "=" * 60)
    print(f"✓ Completed!")
    print(f"  Total samples processed: {len(results)}")
    print(f"  Results saved to: {results_file}")
    print(f"  Attention maps saved to: {output_dir}")
    print(f"\n📊 Next steps - Visualize attention rollout:")
    print(f"\n  Option 1 - Visualize a single sample:")
    print(f"     python compute_attention_rollout.py --save_dir {output_dir}/0/ --show_evolution")
    print(f"     # Output: rollout_outputs/0/attention_rollout.png")
    print(f"\n  Option 2 - Visualize all samples:")
    print(f"     python visualize_all_mmbench.py")
    print(f"     # Or: bash visualize_all_mmbench.sh")
    print(f"\n  Option 3 - Visualize specific range:")
    print(f"     python visualize_all_mmbench.py --start 0 --max-samples 10")
    print("=" * 60)

if __name__ == "__main__":
    main()
