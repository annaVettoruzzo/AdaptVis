"""
Compute and visualize attention rollout for AdaptVis paper Figure 5.

This script loads the per-layer attention maps saved by the model and computes
attention rollout by multiplying attention matrices across layers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from PIL import Image
import re


def load_attention_maps(save_dir):
    """
    Load attention maps from all layers.

    IMPORTANT: The saved attention maps are pre-softmax logits, so we need to
    apply softmax along the last dimension to get proper probabilities.

    Args:
        save_dir: Directory containing diff_*.npy files

    Returns:
        dict: {layer_idx: {'attention': attn, 'start_idx': s, 'end_idx': e}}
    """
    attention_maps = {}
    pattern = re.compile(r'diff_(\d+)_start45_end620\.npy')

    for file in sorted(os.listdir(save_dir)):
        match = pattern.match(file)
        if match:
            layer_idx = int(match.group(1))

            attn = np.load(os.path.join(save_dir, file))

            # CRITICAL FIX: Apply softmax to convert logits to probabilities
            # The saved values are pre-softmax with large negative values from masking
            import torch
            attn_tensor = torch.from_numpy(attn).float()
            attn_softmax = torch.softmax(attn_tensor, dim=-1)
            attn = attn_softmax.numpy()

            attention_maps[layer_idx] = {
                'attention': attn,
                'start_idx': 45,
                'end_idx': 620
            }
            print(f"Loaded layer {layer_idx}: shape={attn.shape}, "
                  f"min={attn.min():.4f}, max={attn.max():.4f}, mean={attn.mean():.4f}")

    return attention_maps


def compute_attention_rollout(attention_maps, image_token_range, num_layers=32):
    """
    Compute attention rollout by averaging last token attention across layers.

    NOTE: Since we only saved the last token's attention (not full attention matrices),
    we cannot compute true attention rollout. Instead, we average the last token's
    attention to image patches across layers.

    Args:
        attention_maps: dict of {layer_idx: {'attention': attn, 'start_idx': s, 'end_idx': e}}
        image_token_range: tuple (start_idx, end_idx) for image tokens
        num_layers: number of layers to use

    Returns:
        np.ndarray: averaged attention to image patches
    """
    start_idx, end_idx = image_token_range

    # Collect attention from each layer
    layer_attentions = []

    for layer_idx in range(min(num_layers, len(attention_maps))):
        if layer_idx not in attention_maps:
            print(f"Warning: Layer {layer_idx} not found, skipping")
            continue

        # Get attention weights for this layer
        # Shape: (batch, num_heads, seq_len) - only last token's attention
        attn = attention_maps[layer_idx]['attention']

        # Average over batch and heads
        # Take mean across batch (dim 0) and heads (dim 1)
        attn_avg = attn.mean(axis=(0, 1))  # (seq_len,)

        # Extract attention to image patches
        image_attn = attn_avg[start_idx:end_idx+1]
        layer_attentions.append(image_attn)

        if layer_idx % 5 == 0:
            print(f"Processed layer {layer_idx}, image attention shape: {image_attn.shape}")

    # Average across all layers
    rollout = np.mean(layer_attentions, axis=0)

    return rollout


def extract_image_attention(rollout, image_token_range, query_token_idx=-1):
    """
    Extract attention to image patches from rollout.

    NOTE: Since rollout is already averaged attention to image patches,
    this just returns the rollout array.

    Args:
        rollout: averaged attention to image patches
        image_token_range: tuple (start_idx, end_idx) for image tokens (unused)
        query_token_idx: index of query token (unused, always last token)

    Returns:
        np.ndarray: attention weights to image patches
    """
    # Rollout is already the attention to image patches
    return rollout


def reshape_to_grid(image_attention, grid_size=24):
    """
    Reshape flattened image attention to 2D grid.

    Args:
        image_attention: flattened attention array (should be 576 for 24x24 grid)
        grid_size: size of grid (assuming square image patches)

    Returns:
        np.ndarray: 2D attention grid
    """
    num_patches = len(image_attention)

    # For CLIP ViT-L/14 in LLaVA 1.5:
    # Input: 336x336 pixels, Patch size: 14x14
    # Grid: 24x24 = 576 patches
    expected_patches = 576

    if num_patches != expected_patches:
        print(f"Warning: Expected {expected_patches} patches (24x24), got {num_patches}")
        if num_patches != grid_size * grid_size:
            # Try to find the closest grid size
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size != num_patches:
                print(f"Warning: {num_patches} patches cannot form perfect square grid. "
                      f"Using {grid_size}x{grid_size} with padding.")
                # Pad to make square
                target_size = (grid_size + 1) * (grid_size + 1)
                padded = np.zeros(target_size)
                padded[:num_patches] = image_attention
                image_attention = padded
                grid_size = grid_size + 1

    # Reshape to grid
    attention_grid = image_attention.reshape(grid_size, grid_size)

    return attention_grid


def visualize_attention_rollout(attention_grid, image_path=None, save_path=None,
                                title="Attention Rollout"):
    """
    Visualize attention rollout as heatmap.

    Args:
        attention_grid: 2D attention grid
        image_path: path to original image (optional)
        save_path: path to save visualization
        title: plot title
    """
    fig, axes = plt.subplots(1, 2 if image_path else 1,
                             figsize=(12, 6) if image_path else (8, 8))

    if image_path:
        # Load and display original image
        img = Image.open(image_path)
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Display attention heatmap
        im = axes[1].imshow(attention_grid, cmap='hot', interpolation='nearest')
        axes[1].set_title(title)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        # Display only attention heatmap
        im = axes.imshow(attention_grid, cmap='hot', interpolation='nearest')
        axes.set_title(title)
        axes.axis('off')
        plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def compute_rollout_per_layer(attention_maps, image_token_range, max_layer=32):
    """
    Compute attention per layer (not cumulative rollout, since we only have last token).

    This shows how attention evolves across layers (useful for multi-panel figures).

    Args:
        attention_maps: dict of layer attention maps
        image_token_range: tuple (start_idx, end_idx)
        max_layer: maximum layer to compute

    Returns:
        dict: {layer_idx: image_attention_array}
    """
    start_idx, end_idx = image_token_range
    attentions = {}

    for layer_idx in range(min(max_layer, len(attention_maps))):
        if layer_idx not in attention_maps:
            continue

        attn = attention_maps[layer_idx]['attention']
        attn_avg = attn.mean(axis=(0, 1))  # Average over batch and heads

        # Extract image attention
        image_attn = attn_avg[start_idx:end_idx+1]
        attentions[layer_idx] = image_attn

    return attentions


def visualize_attention_evolution(rollouts, image_token_range, save_path=None):
    """
    Visualize how attention evolves across layers (multi-panel figure).

    Args:
        rollouts: dict of {layer_idx: image_attention_array}
        image_token_range: tuple (start_idx, end_idx)
        save_path: path to save figure
    """
    # Select layers to visualize (e.g., layers 0, 8, 16, 24, 31)
    layers_to_show = [0, 8, 16, 24, 31]
    layers_to_show = [l for l in layers_to_show if l in rollouts]

    fig, axes = plt.subplots(1, len(layers_to_show), figsize=(4*len(layers_to_show), 4))

    for ax, layer_idx in zip(axes, layers_to_show):
        image_attn = rollouts[layer_idx]

        # Normalize this layer's attention
        image_attn_norm = (image_attn - image_attn.min()) / (image_attn.max() - image_attn.min() + 1e-8)

        attention_grid = reshape_to_grid(image_attn_norm)

        im = ax.imshow(attention_grid, cmap='hot', interpolation='nearest')
        ax.set_title(f"Layer {layer_idx}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Last Token Attention Evolution Across Layers")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved evolution plot to {save_path}")

    plt.show()


def find_image_path_from_save_dir(save_dir):
    """
    Automatically find the original image path based on the save directory.

    Args:
        save_dir: Path like output/Controlled_Images_A_weight0.80/0/

    Returns:
        str: Full path to the original image, or None if not found
    """
    import json

    try:
        # Extract folder number from save_dir
        # Expected format: output/Controlled_Images_A_weight0.80/0/
        path_parts = save_dir.strip('/').split('/')
        if len(path_parts) < 2:
            return None

        folder_name = path_parts[-1]  # e.g., "0"

        # Try to convert to integer
        try:
            folder_num = int(folder_name)
        except ValueError:
            print(f"Warning: Could not parse folder number from: {folder_name}")
            return None

        # Extract output directory and dataset name from path
        # Expected: output/Controlled_Images_A_weight0.80
        if len(path_parts) >= 3:
            output_base = path_parts[-3]  # "output"
            dataset_folder = path_parts[-2]  # "Controlled_Images_A_weight0.80"

            # Extract dataset name (before _weight)
            dataset_name = dataset_folder.split('_weight')[0]

            # Try to load sampled indices
            sampled_idx_file = f"{output_base}/sampled_idx_{dataset_name}.npy"
            if not os.path.exists(sampled_idx_file):
                # Try without output_base
                sampled_idx_file = f"output/sampled_idx_{dataset_name}.npy"
                if not os.path.exists(sampled_idx_file):
                    print(f"Warning: Sampled indices file not found: {sampled_idx_file}")
                    return None

            sampled_indices = np.load(sampled_idx_file)

            if folder_num >= len(sampled_indices):
                print(f"Warning: Folder {folder_num} is out of range (max: {len(sampled_indices)-1})")
                return None

            # Get original dataset index
            original_idx = sampled_indices[folder_num]

            # Load dataset file
            dataset_file = 'data/controlled_images_dataset.json'
            if not os.path.exists(dataset_file):
                print(f"Warning: Dataset file not found: {dataset_file}")
                return None

            with open(dataset_file, 'r') as f:
                dataset = json.load(f)

            if original_idx >= len(dataset):
                print(f"Warning: Original index {original_idx} is out of range")
                return None

            entry = dataset[original_idx]
            image_path = entry['image_path']

            # Make absolute path
            if not os.path.isabs(image_path):
                # Try relative to current directory
                full_path = os.path.join(os.getcwd(), image_path)
                if not os.path.exists(full_path):
                    # Try relative to script directory
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    full_path = os.path.join(script_dir, image_path)

            if os.path.exists(full_path):
                print(f"\n✓ Automatically found image:")
                print(f"  Folder {folder_num} → Original index {original_idx}")
                print(f"  Image: {image_path}")
                print(f"  Caption: {entry['caption_options'][0]}")
                return full_path
            else:
                print(f"Warning: Image file not found: {full_path}")
                return image_path  # Return path anyway, might work

        return None

    except Exception as e:
        print(f"Warning: Error finding image path automatically: {e}")
        return None


def main():
    """
    Main function to compute and visualize attention rollout.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Compute attention rollout')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory containing saved attention maps')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to original image (optional, auto-detected if not provided)')
    parser.add_argument('--output_dir', type=str, default='./rollout_outputs',
                       help='Directory to save outputs')
    parser.add_argument('--num_layers', type=int, default=32,
                       help='Number of layers to use for rollout')
    parser.add_argument('--grid_size', type=int, default=24,
                       help='Grid size for image patches')
    parser.add_argument('--show_evolution', action='store_true',
                       help='Show attention evolution across layers')

    args = parser.parse_args()

    # Extract folder number from save_dir for organizing outputs
    # Expected format: output/Controlled_Images_A_weight0.80/0/
    folder_number = None
    path_parts = args.save_dir.strip('/').split('/')
    if len(path_parts) >= 1:
        try:
            folder_number = int(path_parts[-1])
        except ValueError:
            pass  # Not a numbered folder, use root output_dir

    # Create output directory
    if folder_number is not None:
        # Organize outputs by folder number: rollout_outputs/0/, rollout_outputs/1/, etc.
        actual_output_dir = os.path.join(args.output_dir, str(folder_number))
    else:
        actual_output_dir = args.output_dir

    os.makedirs(actual_output_dir, exist_ok=True)

    # Auto-detect image path if not provided
    if args.image_path is None:
        print("\nNo image_path provided, attempting auto-detection...")
        args.image_path = find_image_path_from_save_dir(args.save_dir)
        if args.image_path is None:
            print("\nCould not auto-detect image path. Proceeding without original image overlay.")
        else:
            print()  # Empty line for spacing

    # Load attention maps
    print(f"Loading attention maps from {args.save_dir}...")
    attention_maps = load_attention_maps(args.save_dir)

    if not attention_maps:
        print("No attention maps found!")
        return

    # Get image token range from first layer
    first_layer = list(attention_maps.values())[0]
    image_token_range = (first_layer['start_idx'], first_layer['end_idx'])
    print(f"Image token range: {image_token_range}")

    # Compute attention rollout
    print("\nComputing attention rollout...")
    rollout = compute_attention_rollout(attention_maps, image_token_range, args.num_layers)

    # Extract image attention (rollout is already image attention)
    image_attention = extract_image_attention(rollout, image_token_range)
    print(f"Image attention shape: {image_attention.shape}")

    # Normalize to 0-1 range for better visualization
    image_attention_norm = (image_attention - image_attention.min()) / (image_attention.max() - image_attention.min() + 1e-8)

    # Reshape to grid
    attention_grid = reshape_to_grid(image_attention_norm, args.grid_size)
    print(f"Attention grid shape: {attention_grid.shape}")

    # Visualize
    save_path = os.path.join(actual_output_dir, 'attention_rollout.png')
    print(f"Saving visualization to: {save_path}")
    visualize_attention_rollout(attention_grid, args.image_path, save_path)

    # Optionally show evolution
    if args.show_evolution:
        print("\nComputing attention evolution...")
        attentions = compute_rollout_per_layer(attention_maps, image_token_range, args.num_layers)
        evolution_path = os.path.join(actual_output_dir, 'attention_evolution.png')
        print(f"Saving evolution plot to: {evolution_path}")
        visualize_attention_evolution(attentions, image_token_range, evolution_path)

    print(f"\nDone! Outputs saved to: {actual_output_dir}")


if __name__ == "__main__":
    """
    Example usage:
    # Auto-detect image path and organize outputs by folder number
    python compute_attention_rollout.py --save_dir ./output/Controlled_Images_A_weight0.80/0/ --show_evolution
    # Outputs will be saved to: ./rollout_outputs/0/

    # Specify image path manually
    python compute_attention_rollout.py --save_dir ./output/Controlled_Images_A_weight0.80/5/ --image_path data/controlled_images/cap_under_chair.jpeg
    # Outputs will be saved to: ./rollout_outputs/5/
    """
    main()
