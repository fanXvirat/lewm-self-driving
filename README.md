<p align="center">
  <h1 align="center">LeWM Self-Driving</h1>
  <p align="center">
    <a href="https://arxiv.org/abs/2603.19312">Paper</a> &middot;
    <a href="https://github.com/lucas-maes/le-wm">Original Repo</a> &middot;
    <a href="https://huggingface.co/datasets/commaai/comma2k19">Dataset</a>
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#paper-faithfulness">Paper Faithfulness</a> •
  <a href="#results">Results</a>
</p>

---

## Training

The model is trained **end-to-end from raw pixels** using a Joint-Embedding Predictive Architecture (JEPA).

- Dataset: Comma2k19 (789 segments, 4 chunks)
- Hardware: NVIDIA H100 80GB
- Training: 20 epochs
- Decoder: trained separately for 25 epochs (visualization only)

Key properties:
- No reconstruction loss during JEPA training
- No EMA or stop-gradient
- Fully autoregressive latent prediction
- Stable training with SIGReg regularization

---

## Architecture

```text
                    Raw frames o_t
                          │
                          ▼
              ┌─────────────────────┐
              │   ViT Encoder       │
              │   12 layers         │
              │   patch size 14     │
              │   CLS token         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Projection MLP    │
              │   + BatchNorm       │
              └──────────┬──────────┘
                         │
                         ▼
                   latent z_t (384-dim)
                    │            │
                    │            ▼
                    │     ┌─────────────┐
                    │     │   SIGReg    │
                    │     │ (anti-      │
                    │     │  collapse)  │
                    │     └─────────────┘
                    ▼
        ┌───────────────────────────┐
        │  Causal Transformer       │
        │  Predictor                │
        │  8 layers · 16 heads      │
        │  AdaLN action conditioning│
        └──────────┬────────────────┘
                   ▲
        ┌──────────┴────────────────┐
        │  Action Embedder          │
        │  (steer, speed, brake)    │
        │  × frameskip=5            │
        │  Conv1D + MLP             │
        └───────────────────────────┘
                   │
                   ▼
           predicted ẑ_(t+1)

Training loss:
  L = MSE(ẑ_(t+1), z_(t+1))
    + λ · SIGReg(Z)
    + repr_cosine_smoothness
````

**Total parameters: 57.7M**

---

## Paper Faithfulness

Every core component maps directly to the LeWorldModel paper.
Driving-specific elements are clean extensions, not modifications.

| Paper Component         | Paper Specification             | This Implementation                    |
| :---------------------- | :------------------------------ | :------------------------------------- |
| **Training paradigm**   | End-to-end JEPA from raw pixels | End-to-end from raw front-camera video |
| **Encoder**             | Vision Transformer (ViT)        | ViT via Hugging Face `ViTModel`        |
| **Patch size**          | 14                              | 14                                     |
| **Latent extraction**   | Last-layer CLS token            | Last-layer CLS token                   |
| **Projection head**     | 1-layer MLP + BatchNorm         | `ProjMLP` + `SafeBN1d`                 |
| **Predictor**           | Causal Transformer              | 8-layer causal Transformer             |
| **Action conditioning** | AdaLN                           | 6-way AdaLN-zero (`CondBlock`)         |
| **AdaLN init**          | Zero init                       | Zero init                              |
| **Prediction target**   | z_(t+1)                         | z_(t+1)                                |
| **Main loss**           | MSE                             | MSE                                    |
| **Anti-collapse**       | SIGReg                          | SIGReg (1024 projections)              |
| **EMA / stop-grad**     | Not used                        | Not used                               |
| **Pretrained encoder**  | Not required                    | Not used                               |
| **Reconstruction loss** | Not used                        | Not used                               |
| **Planning**            | CEM                             | CEM (300 samples, 30 iters)            |
| **Decoder**             | Visualization only              | Trained separately                     |

### Driving-specific extensions

| Addition         | Description                            |
| :--------------- | :------------------------------------- |
| Driving actions  | (steer, speed, brake) with frameskip=5 |
| Smoothness loss  | Small cosine temporal regularization   |
| Larger latent    | 384-dim vs 192                         |
| Decoder training | Fully separate from JEPA               |

---

## Results

### Open-loop prediction (BM1)

The model predicts future latent states at 10x lower error than a null baseline:

![BM1](docs/assets/BM1_openloop.png)

### Action sensitivity (BM2)

Strong response to control inputs (accel vs brake divergence = 1.17):

![BM2](docs/assets/BM2_action_sensitivity.png)

### CEM planner (BM3)

Latent-space planning outperforms random search:

![BM3](docs/assets/BM3_cem.png)

### Decoder reconstructions (BM4)

Structurally correct outputs; blur due to decoder limitation:

![BM4](docs/assets/BM4_visual.png)

### Latent structure (T2)

Clean clustering by driving state:

![T2](docs/assets/t2_latent_tsne.png)

### Embedding health (T8)

* 0 collapsed dims
* Effective rank: 34.9 / 384

![T8](docs/assets/t8_embedding_health.png)

### Future selection (VL1)

100% accuracy selecting correct future:

![VL1](docs/assets/VL1_choose_correct_future.png)

### Training curves (T0)

Stable convergence:

![T0](docs/assets/t0_training_curves.png)

### Temporal straightening (T9)

Emergent trajectory structure:

![T9](docs/assets/t9_temporal.png)

### Planning rollout (tA)

CEM-guided imagination:

![tA](docs/assets/tA_rollout.png)

### Retrieval (tB)

Semantic nearest neighbors:

![tB](docs/assets/tB_retrieval.png)

### Summary panel (tE)

Final consolidated metrics:

![tE](docs/assets/tE_summary.png)

---

## Quick Start

### Install

```bash
pip install torch torchvision transformers einops matplotlib scipy scikit-learn av huggingface_hub
```

### Train

```bash
python train.py --lightning --chunks 1 2 3 4 --budget-hours 2.9
```

or

```bash
python train.py --data-dir ./comma2k19 --chunks 1 2 --budget-hours 3.0
```

---

### Evaluate

```bash
python tests.py --checkpoint ./checkpoints_v5/final.pt --lightning --output-dir ./results
```

---

## Hardware

* Trained on NVIDIA H100 80GB
* bf16 mixed precision + torch.compile
* ~84GB total VRAM usage

Minimum: 24GB VRAM (auto-adjusts batch size)

---

## Dataset

Comma2k19 — 33 hours of driving with:

* Front camera video
* CAN bus (speed, steering)
* IMU signals

Supports:

* NVDEC GPU decoding (fast)
* PyAV CPU fallback

---

## Reference

```
@article{maes2026leworldmodel,
  title   = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author  = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal = {arXiv:2603.19312},
  year    = {2026}
}
```

```
