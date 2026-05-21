# 4KLSDB — Model Submodules

This directory hosts every model used in the 4KLSDB paper as a git submodule
pinned to the exact upstream commit we trained against. After
`git clone --recurse-submodules https://github.com/taco-group/4KLSDB.git`
you should see:

| Folder              | Upstream                                                   | Used for                                 | Wrapper scripts                                                                          |
|---------------------|------------------------------------------------------------|------------------------------------------|------------------------------------------------------------------------------------------|
| `hit_sr/`           | https://github.com/XiangZ-0/HiT-SR                         | classical SR ×4 / cascade ×8 / ×16       | `../scripts/train_hit_sr.sh`, `../scripts/inference_classical_sr.sh --model hit_sr`      |
| `swinir/`           | https://github.com/JingyunLiang/SwinIR                     | classical SR + GAN-based finetune        | `../scripts/train_swinir.sh`, `../scripts/inference_classical_sr.sh --model swinir`      |
| `mambair/`          | https://github.com/csguoh/MambaIR                          | classical SR ×4 / ×8 / ×16               | `../scripts/train_mambair.sh`, `../scripts/inference_classical_sr.sh --model mambair`    |
| `osediff/`          | https://github.com/cswry/OSEDiff                           | real-world blind SR (one-step diffusion) | `../scripts/train_osediff.sh`, `../scripts/inference_real_sr.sh --model osediff`         |
| `seesr/`            | https://github.com/cswry/SeeSR                             | real-world SR with semantics             | `../scripts/train_seesr.sh`,   `../scripts/inference_real_sr.sh --model seesr`           |
| `sana/`             | https://github.com/NVlabs/Sana                             | 4K text-to-image generation              | `../scripts/train_sana_4k.sh`, `../scripts/inference_sana_4k.sh`                         |

## 4KLSDB-specific patches

For each submodule we add a small `4klsdb/` patch directory at the root with:

* **YAML / JSON configs** for the BasicSR-style models that point at the 4KLSDB train/val/test splits and the cropping logic described in §4 of the paper.
* **A blind degradation pipeline** (scale-guided hyper-network) for OSEDiff & SeeSR (§4.2 of the paper).
* **A 4096²-resolution config** for Sana plus the `embed_pro.py` Gemma-2 embedding pre-computer (see Section 1 below).

## Getting checkpoints

```bash
bash ../scripts/download_all_ckpts.sh        # → ../release_ckpts/<model>/
```

This pulls every 4KLSDB-fine-tuned checkpoint from
`https://huggingface.co/taco-group/4KLSDB-<model>`.

---

# Section 1: SANA Model

# SANA 1.5: Preprocessing and Training with Custom WebDataset

This guide explains how to preprocess and train SANA 1.5 using your own high-resolution images and captions, leveraging your custom WebDataset pipeline.

## 1. Environment Setup

First, activate the SANA environment. You can build the environment either by:

- **Using the provided YAML file:**
  ```bash
  conda env create -f 4KLSDB/envs/Sana_training.yml
  conda activate Sana
  ```
- **Or follow the official SANA repo instructions.**

## 2. Preprocessing: Creating a SANA-Compatible WebDataset

Your dataset preprocessing is handled by `embed_pro.py`, which takes your high-resolution images and captions and produces a multi-shard WebDataset (including latents, text embeddings, and metadata) ready for SANA 1.5 training.

### Usage Example

```bash
cd 4KLSDB/models/sana/diffusion/data/datasets

# Multi-GPU example (replace CUDA_VISIBLE_DEVICES as needed)
CUDA_VISIBLE_DEVICES=4,5,6 \
torchrun --nproc_per_node=3 --master_port=29500 \
  embed_pro.py \
  --img-dir /path/to/your/HR_images \
  --webdataset-dir /path/to/output/webdataset \
  --tile-size 1024 \
  --tile-overlap 128 \
  --scale 0.41407 \
  --batch-size 16 \
  --num-workers 1 \
  --precision bf16 \
  --text-batch-size 16 \
  --checkpoint-every 100 \
  --resume-from-checkpoint
```

**Key arguments:**
- `--img-dir`: Directory containing your high-resolution images (with matching `.txt` caption files).
- `--webdataset-dir`: Output directory for the generated WebDataset shards and `wids-meta.json`.
- `--tile-size`, `--tile-overlap`, `--scale`: Control image tiling and latent scaling.
- `--batch-size`, `--text-batch-size`: Batch sizes for image and text processing.
- `--precision`: Use `bf16` for optimal speed/VRAM if supported.
- `--checkpoint-every`: Frequency of checkpointing.
- `--resume-from-checkpoint`: Resume from last checkpoint if interrupted.

After completion, your output directory will contain:
- Sharded `.tar` files with latents, text embeddings, and metadata.
- A `wids-meta.json` file describing the dataset and shards.

## 3. Training SANA 1.5 with Your WebDataset

Once preprocessing is complete, you can train SANA 1.5 directly from your WebDataset.

### Training Command

```bash
cd 4KLSDB/models/sana

# Single GPU
python train.py \
  --config configs/sana1.5/your_config.yaml \
  --data_path /path/to/output/webdataset/wids-meta.json \
  --output_dir /path/to/training_output \
  --mixed_precision bf16

# Multi-GPU (recommended for large datasets)
torchrun --nproc_per_node=NUM_GPUS train.py \
  --config configs/sana1.5/your_config.yaml \
  --data_path /path/to/output/webdataset/wids-meta.json \
  --output_dir /path/to/training_output \
  --mixed_precision bf16
```

**Replace:**
- `/path/to/output/webdataset/wids-meta.json` with the path to your generated metadata file.
- `NUM_GPUS` with the number of GPUs you wish to use.

**Notes:**
- The training script will automatically use your WebDataset for efficient multi-scale training.
- Make sure your `your_config.yaml` is set up for WebDataset input (`data.type: SanaWebDatasetMS`).

## 4. Tips

- Your WebDataset must have the correct structure: each sample includes a latent (`npy`), text embedding (`npz`), and metadata (`json`).
- The `wids-meta.json` file is required for training and is generated by your preprocessing script.
- For best performance, store your WebDataset on fast local storage (SSD).
