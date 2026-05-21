# 4KLSDB — Scripts

One-click wrappers around every model used in the paper. All scripts assume
you've cloned this repo **with submodules** (`git clone --recurse-submodules
https://github.com/taco-group/4KLSDB.git`) and have the 4KLSDB-fine-tuned
checkpoints under `release_ckpts/` (see `download_all_ckpts.sh`).

## Download

| Script                       | What it does                                                              |
|------------------------------|---------------------------------------------------------------------------|
| `download_all_ckpts.sh`      | Pull every fine-tuned checkpoint from `taco-group/4KLSDB-*` on 🤗 into `release_ckpts/<model>/`. |

## Inference

| Script                          | Models supported                       | Output                       |
|---------------------------------|----------------------------------------|------------------------------|
| `inference_classical_sr.sh`     | HiT-SR, SwinIR, MambaIR (×4 / ×8 / ×16) | upscaled images              |
| `inference_real_sr.sh`          | OSEDiff, SeeSR (×4 / ×8 / ×16)          | upscaled images              |
| `inference_sana_4k.sh`          | Sana (4096² T2I)                        | one 4K image per prompt      |

Every script has `--help`.

## Training

| Script              | Model     | Notes                                          |
|---------------------|-----------|------------------------------------------------|
| `train_hit_sr.sh`   | HiT-SR    | BasicSR-style YAML config                      |
| `train_swinir.sh`   | SwinIR    | KAIR-style JSON config                         |
| `train_mambair.sh`  | MambaIR   | BasicSR-style YAML config                      |
| `train_osediff.sh`  | OSEDiff   | Accelerate / Diffusers, blind degradation      |
| `train_seesr.sh`    | SeeSR     | Accelerate / Diffusers + ControlNet            |
| `train_sana_4k.sh`  | Sana      | 4096² fine-tune; pre-compute Gemma-2 embeddings first |

Multi-GPU launches: prefix with `NPROC=8` (or pass `--nproc 8`) on the BasicSR
scripts. The diffusers-based scripts use `accelerate launch --num_processes`
internally.
