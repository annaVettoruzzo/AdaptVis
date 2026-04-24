# Computing Attention Rollout for AdaptVis

This document explains how to compute and visualize attention rollout from saved attention maps.

## Overview

The AdaptVis model saves per-layer attention maps during inference. These are used to compute **attention rollout**, showing how attention accumulates across layers to focus on specific image regions.

**Three workflows available:**
1. **MMBench Dataset** (VQA benchmark with 20 categories)
2. **POPE Dataset** (object hallucination evaluation, 3 categories)
3. **Controlled Images Dataset** (original research setup)

## Quick Start

### **MMBench (VQA):**
```bash
# Step 1: Generate attention maps (10 samples per category)
python rollout_mmbench.py --n-samples 10

# Step 2: Visualize all samples
python visualize_all_mmbench.py
```

### **POPE (Object Hallucination):**
```bash
# Step 1: Generate attention maps (10 samples per category)
python rollout_pope.py --n-samples 10

# Step 2: Visualize all samples
python visualize_all_pope.py
```

### **Controlled Images:**
```bash
# Step 1: Run model with attention saving
python3 main_aro.py --dataset=Controlled_Images_A --model-name='llava1.5' \
    --download --method=scaling_vis --weight=0.8 --option=four

# Step 2: Visualize attention rollout
python compute_attention_rollout.py \
    --save_dir ./output/Controlled_Images_A_weight0.80/0/ --show_evolution
```

---

## MMBench Workflow

### Step 1: Generate Attention Maps

```bash
# All categories, 100 samples each
python rollout_mmbench.py

# Specific categories
python rollout_mmbench.py --groups-filter "action_recognition,object_localization" --n-samples 50

# Test with 10 samples
python rollout_mmbench.py --n-samples 10
```

**Output:** `./output/mmbench_base/{idx}/`
- `diff_*.npy` - Attention maps per layer
- `original_image.png` - Original image for overlay
- `results.json` - Prompts, generations, ground truth

**Available Categories:**
```
action_recognition, attribute_comparison, attribute_recognition,
celebrity_recognition, function_reasoning, identity_reasoning,
image_emotion, image_quality, image_scene, image_style,
image_topic, nature_relation, object_localization, ocr,
physical_property_reasoning, physical_relation, social_relation,
spatial_relationship, structuralized_imagetext_understanding,
future_prediction
```

### Step 2: Visualize

**Single sample:**
```bash
python compute_attention_rollout.py --save_dir ./output/mmbench_base/0/ --show_evolution
```

**Batch processing:**
```bash
# All samples
python visualize_all_mmbench.py

# First 10 samples
python visualize_all_mmbench.py --max-samples 10

# Samples 10-20
python visualize_all_mmbench.py --start 10 --max-samples 10
```

**Output:** `./rollout_outputs/mmbench/{idx}/`
- `attention_rollout.png` - Attention heatmap with image overlay
- `attention_evolution.png` - Layer-by-layer evolution
- `sample_info.json` - Prompt, generation, golden answer, category

---

## POPE Workflow

### Step 1: Generate Attention Maps

```bash
# All categories, 100 samples each
python rollout_pope.py

# Specific categories
python rollout_pope.py --groups-filter "adversarial,popular" --n-samples 50

# Test with 10 samples
python rollout_pope.py --n-samples 10
```

**Output:** `./output/pope_base/{idx}/`
- `diff_*.npy` - Attention maps per layer
- `original_image.png` - Original image for overlay
- `results.json` - Questions, generations, ground truth

**Available Categories:**
```
adversarial, popular, random
```

### Step 2: Visualize

**Single sample:**
```bash
python compute_attention_rollout.py --save_dir ./output/pope_base/0/ --show_evolution
```

**Batch processing:**
```bash
# All samples
python visualize_all_pope.py

# First 10 samples
python visualize_all_pope.py --max-samples 10

# Samples 10-20
python visualize_all_pope.py --start 10 --max-samples 10
```

**Output:** `./rollout_outputs/pope/{idx}/`
- `attention_rollout.png` - Attention heatmap with image overlay
- `attention_evolution.png` - Layer-by-layer evolution
- `sample_info.json` - Question, generation, golden answer, category

---

## Controlled Images Workflow

### Step 1: Run Model

```bash
# Setup environment
mkdir data output outputs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run model with attention saving
python3 main_aro.py --dataset=Controlled_Images_A --model-name='llava1.5' \
    --download --method=scaling_vis --weight=0.8 --option=four
```

**Output:** `./output/Controlled_Images_A_weight0.80/{idx}/`
- `diff_*.npy` - Attention maps (pre-softmax logits)

### Step 2: Visualize

```bash
# Auto-detects image path
python compute_attention_rollout.py \
    --save_dir ./output/Controlled_Images_A_weight0.80/0/ --show_evolution
```

---

## Parameters

### `rollout_mmbench.py`
- `--n-samples`: Samples per category (default: 10)
- `--groups-filter`: Comma-separated categories
- `--output-dir`: Output directory (default: `./output/mmbench_base`)
- `--device`: CUDA device (default: `cuda`)

### `rollout_pope.py`
- `--n-samples`: Samples per category (default: 10)
- `--groups-filter`: Comma-separated categories (adversarial, popular, random)
- `--output-dir`: Output directory (default: `./output/pope_base`)
- `--device`: CUDA device (default: `cuda`)

### `compute_attention_rollout.py`
- `--save_dir`: Directory with attention maps (required)
- `--image_path`: Original image path (optional, auto-detected)
- `--output_dir`: Output directory (default: `./rollout_outputs`)
- `--interpolation`: `nearest` (patches), `bilinear` (smooth), `bicubic`, `spline16`
- `--show_evolution`: Show layer-by-layer evolution
- `--num_layers`: Layers to use (default: 32)
- `--grid_size`: Patch grid size (default: 24)

### `visualize_all_mmbench.py`
- `--max-samples`: Limit samples to process
- `--start`: Start from specific index
- `--no-show-evolution`: Skip evolution plots (faster)
- `--base-dir`: Input directory (default: `./output/mmbench_base`)
- `--output-dir`: Output directory (default: `./rollout_outputs/mmbench`)

### `visualize_all_pope.py`
- `--max-samples`: Limit samples to process
- `--start`: Start from specific index
- `--no-show-evolution`: Skip evolution plots (faster)
- `--base-dir`: Input directory (default: `./output/pope_base`)
- `--output-dir`: Output directory (default: `./rollout_outputs/pope`)

---

## Understanding the Output

### Attention Rollout Heatmap
- **Brighter colors** (yellow/white) = higher attention
- **Darker colors** (blue/black) = lower attention
- **All 576 patches** shown (24×24 grid for LLaVA 1.5)

### Patch-based Visualization
- LLaVA 1.5 processes images as **24×24 patches** (576 total)
- Each patch covers **14×14 pixels** of the 336×336 image
- `--interpolation nearest`: Shows discrete patches (recommended)
- `--interpolation bilinear`: Smooth interpolation

**Why only some patches appear highlighted:**
The model learns to focus on relevant regions. Most patches have low attention (blue), few have high attention (red/yellow). This is expected behavior.

### Attention Evolution
Shows how attention changes across layers:
- **Early layers** (0, 8): Diffuse attention
- **Middle layers** (16): Starts focusing
- **Late layers** (24, 31): Sharp focus on relevant regions

### MMBench Sample Information
```json
{
  "Prompt": "USER: <image>\nWhich one is correct?\n...",
  "Generation": "B",
  "Golden": "B",
  "Category": "action_recognition"
}
```

### POPE Sample Information
```json
{
  "Prompt": "USER: <image>\nIs there a cat in the image?\nASSISTANT:",
  "Generation": "yes",
  "Golden": "yes",
  "Category": "adversarial"
}
```

---

## Technical Details

### Attention Rollout Computation
Since we only save the last token's attention, we average across layers:
```
Rollout = mean(Layer_0_attention, Layer_1_attention, ..., Layer_31_attention)
```

### Pre-softmax Logits
Saved attention maps are pre-softmax logits. The script applies softmax automatically to convert to probabilities.

### Image Token Mapping (LLaVA 1.5)
- Input: 336×336 pixels
- Patch size: 14×14 pixels
- Grid: 24×24 = 576 patches
- Token positions: 5-580 (image tokens)

