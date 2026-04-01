# Computing Attention Rollout for AdaptVis

This document explains how to compute and visualize attention rollout (like Figure 5 in the paper) from the saved attention maps.

## Overview

The AdaptVis model saves per-layer attention maps during inference. These can be used to compute **attention rollout**, which shows how attention accumulates across layers to focus on specific image regions.

## Quick Start

### Step 1: Run Model to Save Attention Maps

First, run the model with attention saving enabled. The attention maps are saved automatically when using the ADAPTVIS or ScalingVis methods.

Following the original README.md file run the following commands to set up the environment.
```bash
mkdir data
mkdir output
mkdir outputs
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the model and download the data (attention maps are saved automatically).
```bash 
python3 main_aro.py --dataset=Controlled_Images_A --model-name='llava1.5' --download --method=base  --option=four
```

Attention maps will be saved to:
```
./output/{dataset}_weight{weight}/{index}/
```

Each file is named: `diff_{layer}_start{start}_end{end}.npy`

For example: `diff_0_start576end1151.npy` (layer 0, image tokens 576-1151)

### Step 2: Compute Attention Rollout

**Automatic image detection (recommended)**:

The script can now automatically find the original image path and organizes outputs by folder number:

```bash
# Auto-detects the image and shows it alongside attention heatmap
# Outputs saved to: rollout_outputs/0/
python compute_attention_rollout.py \
    --save_dir ./output/Controlled_Images_A_weight1.00/0/ \
    --show_evolution
```

**Output Organization**:
```
rollout_outputs/
├── 0/                          # Outputs for image 0
│   ├── attention_rollout.png
│   └── attention_evolution.png
├── 1/                          # Outputs for image 1
│   └── attention_rollout.png
└── 5/                          # Outputs for image 5
    └── attention_rollout.png
```

**Manual image path**:

If auto-detection fails or you want to use a different image:

```bash
python compute_attention_rollout.py \
    --save_dir ./output/Controlled_Images_A_weight1.00/0/ \
    --image_path data/controlled_images/wineglass_on_table.jpeg \
    --output_dir ./rollout_outputs
```

## Script Features

### `compute_attention_rollout.py`

Full-featured script with more options:

```bash
python compute_attention_rollout.py \
    --save_dir PATH          # Directory with .npy files \
    --image_path PATH        # Optional: original image \
    --output_dir PATH        # Output directory (default: ./rollout_outputs) \
    --num_layers 32          # Number of layers to use \
    --grid_size 24           # Image patch grid size (default: 24) \
    --show_evolution         # Show attention across layers
```

**Features**:
- Customizable number of layers
- Overlay attention on original image
- Multi-panel figure showing evolution across layers
- Saves high-resolution figures

## Understanding the Output

### Attention Rollout Heatmap

The heatmap shows the model's attention to different image patches:

- **Brighter colors** (yellow/white) = higher attention
- **Darker colors** (red/black) = lower attention

### Important Notes

1. **Pre-softmax logits**: The saved attention maps are pre-softmax (raw logits), so the script applies softmax to convert them to probabilities.

2. **Last token only**: The saved data only contains the last token's attention (not full attention matrices), so we average across layers instead of computing true attention rollout.

3. **Normalization**: The script normalizes each layer's attention to 0-1 range for better visualization.

### Attention Evolution (with `--show_evolution`)

Shows how attention changes across layers:

```
Layer 0  →  Layer 8  →  Layer 16  →  Layer 24  →  Layer 31
(early)                                           (late)
```

Typical pattern:
- **Early layers**: Diffuse attention
- **Middle layers**: Starts focusing
- **Late layers**: Sharp focus on relevant regions

## Technical Details

### Attention Rollout Computation

Attention rollout is computed as:

```
Rollout_l = I + Σ(A_i × Rollout_{i-1})
```

Where:
- `I` = identity matrix (residual connection)
- `A_i` = attention matrix at layer i
- We use: `0.5 × I + 0.5 × A` for each layer

This accumulates attention across all layers, showing the final attention distribution.

### Image Token Mapping

For CLIP ViT-L/14 (used in LLaVA 1.5):
- Input image: 336×336 pixels
- Patch size: 14×14 pixels
- Grid: 24×24 = 576 patches

The attention weights map to these 576 patches, which are reshaped to a 24×24 grid for visualization.

## File Structure

After running the model and computing rollout:

```
output/
├── Controlled_Images_A_weight1.00/
│   ├── 0/                          # Sample index 0
│   │   ├── diff_0_start45end620.npy   # Layer 0
│   │   ├── diff_1_start45end620.npy   # Layer 1
│   │   └── ... (layers 0-31)
│   ├── 1/                          # Sample index 1
│   └── ...
│
└── rollout_outputs/                # Generated visualizations
    ├── 0/                          # Outputs for sample 0
    │   ├── attention_rollout.png
    │   └── attention_evolution.png
    ├── 1/                          # Outputs for sample 1
    └── ...
```

**Attention maps (`.npy` files)**:
- Shape: `(batch_size, num_heads, seq_len)` - last token's attention only
- Pre-softmax logits (script applies softmax automatically)
- Extracted: image tokens at positions 45-620 (576 patches = 24×24 grid)

## References

- Abnar & Zuidema (2020): "Quantifying Attention Flow in Transformers"
- Chefer et al. (2021): "Transformer Interpretability Beyond Attention Visualization"

## Citation

If you use this code, please cite the AdaptVis paper:

```bibtex
@misc{chen2025spatialreasoninghardvlms,
      title={Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas},
      author={Shiqi Chen and Tongyao Zhu and Ruochen Zhou and Jinghan Zhang and Siyang Gao and Juan Carlos Niebles and Mor Geva and Junxian He and Jiajun Wu and Manling Li},
      year={2025},
      eprint={2503.01773},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.01773},
}
```
