#!/usr/bin/env python3
"""
tests.py — Evaluation suite for LeWM Self-Driving

Loads a trained checkpoint and runs the full set of benchmarks:
  BM1: Open-loop L2 error vs null model
  BM2: Action sensitivity (does the latent space respond to different actions?)
  BM3: CEM planner performance
  BM4: Decoder reconstruction quality

  T0:  Training curves
  T1:  Dataset sample visualisation
  T2:  Latent space t-SNE coloured by speed/steering/acceleration
  T3:  Multi-step prediction error (50 sequences)
  T8:  Embedding health (rank, distribution, collapse check)
  T9:  Temporal straightening
  TV1: Frame-to-frame cosine similarity heatmaps
  VL1: Future-selection accuracy (can the model pick the correct future?)

Usage:
  python tests.py --checkpoint ./checkpoints_v5/final.pt --data-dir ./comma2k19
  python tests.py --checkpoint ./checkpoints_v5/final.pt --lightning
"""

import os, sys, math, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
sys.argv = [""]  # prevent train.py main() from running on import
from train import (Config, DrivingLeWM, LatentDecoder,
                             Comma2k19Dataset, _unwrap, lightning_ai_paths)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = Config()
    for k, v in ck["config"].items():
        if hasattr(cfg, k): setattr(cfg, k, v)
    cfg.device = device
    cfg.num_workers = 0

    model = DrivingLeWM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    decoder = None
    if "decoder" in ck:
        decoder = LatentDecoder(cfg.embed_dim)
        decoder.load_state_dict(ck["decoder"])
        decoder.eval()

    hist = ck.get("history", {})
    return model, decoder, cfg, hist


def _denorm(t):
    MN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    SD = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    return (t.cpu().float()*SD+MN).clamp(0,1)


def _to_img(t):
    return _denorm(t).permute(0,2,3,1).numpy()


# ---------------------------------------------------------------------------
# T0: Training curves
# ---------------------------------------------------------------------------

def test_t0_curves(hist, out="t0_training_curves.png"):
    if not hist:
        print("  T0: no history, skipping"); return
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("LeWM V5 Training Curves", fontsize=16, fontweight="bold")
    sm = lambda v, w=50: np.convolve(v, np.ones(w)/w, "valid") if len(v)>=w else v

    plots = [
        ("pred",    axes[0,0], "Pred",    "#2196F3"),
        ("sigreg",  axes[0,1], "Sigreg",  "#F44336"),
        ("lr",      axes[0,2], "LR",      "#FF9800"),
        ("total",   axes[1,0], "Total",   "#4CAF50"),
        ("repr",    axes[1,1], "Repr",    "#9C27B0"),
    ]
    for key, ax, title, col in plots:
        if key in hist and hist[key]:
            v = hist[key]
            ax.plot(v, alpha=0.15, color=col)
            ax.plot(sm(v), color=col, lw=2)
        ax.set_title(title); ax.grid(True, alpha=0.3)

    # Straightening
    ax = axes[1,2]
    if hist.get("straighten"):
        ss, sv = zip(*hist["straighten"])
        ax.plot(ss, sv, 'o-', color="#FF9800", lw=2)
        ax.axhline(0, color="gray", ls="--")
        ax.set_title(f"Temporal Straightening (final={sv[-1]:.3f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  T0 saved: {out}")


# ---------------------------------------------------------------------------
# T1: Dataset samples
# ---------------------------------------------------------------------------

def test_t1_data(cfg, eval_idx, out="t1_data_samples.png"):
    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=50)
    if len(ds) == 0: print("  T1: empty dataset"); return

    n_rows, T = 3, 6
    fig, axes = plt.subplots(n_rows, T, figsize=(T*3, n_rows*3))
    fig.suptitle("Real Comma2k19 Samples", fontsize=14, fontweight="bold")
    for r in range(n_rows):
        frames = _to_img(ds[r * max(len(ds)//n_rows,1)][0])
        step = max(1, len(frames)//T)
        for t in range(T):
            idx = min(t*step, len(frames)-1)
            axes[r,t].imshow(frames[idx])
            axes[r,t].axis("off")
            axes[r,t].set_title(f"t={t}", fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  T1 saved: {out}")


# ---------------------------------------------------------------------------
# T2: Latent space t-SNE
# ---------------------------------------------------------------------------

def test_t2_tsne(model, cfg, eval_idx, n_samples=500, out="t2_tsne.png"):
    from sklearn.manifold import TSNE
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)

    ds = Comma2k19Dataset(eval_idx, cfg, return_state=True, stride=4)
    embs, states = [], []
    with torch.no_grad():
        for i in range(min(n_samples, len(ds))):
            fr, ac, st = ds[i]
            e = raw.model.encode(fr.unsqueeze(0).to(dev)).cpu()
            embs.append(e[0, e.size(1)//2])
            states.append(st[st.size(0)//2])

    E = torch.stack(embs).numpy()
    S = torch.stack(states).numpy()

    proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(E)

    labels = ["Speed(m/s)", "Steer(deg)", "Accel", "GyroZ", "SpeedN", "SteerN"]
    ranges = [(S[:,i].min(), S[:,i].max()) for i in range(min(6, S.shape[1]))]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Latent Space t-SNE ({n_samples} samples, H100)\n"
                 f"Colorbars = ACTUAL signal ranges", fontsize=13, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        if i >= S.shape[1]: break
        lo, hi = ranges[i]
        sc = ax.scatter(proj[:,0], proj[:,1], c=S[:,i], cmap="viridis",
                        s=20, alpha=0.8, vmin=lo, vmax=hi)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"{labels[i]} [{lo:.1f},{hi:.1f}]")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  T2 saved: {out}")
    raw.cpu()


# ---------------------------------------------------------------------------
# T3: Multi-step prediction error
# ---------------------------------------------------------------------------

def test_t3_prediction(model, cfg, eval_idx, n_seqs=50, out="t3_prediction.png"):
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)

    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=6)
    H = cfg.seq_len - 1
    mse_per_step = np.zeros(H)
    emb_traj = []

    with torch.no_grad():
        for i in range(min(n_seqs, len(ds))):
            fr, ac = ds[i]
            emb = raw.model.encode(fr.unsqueeze(0).to(dev))
            ae  = raw.model.encode_actions(ac.unsqueeze(0).to(dev))
            pred = raw.model.predict(emb[:,:-1], ae[:,:-1])
            mse = (pred - emb[:,1:]).pow(2).mean(-1).squeeze().cpu().numpy()
            if mse.ndim == 0: mse = mse[None]
            mse_per_step[:len(mse)] += mse[:H]
            if i < 1: emb_traj = emb.squeeze(0).cpu().numpy()

    mse_per_step /= max(n_seqs, 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"Prediction Quality — {H} steps (mean={mse_per_step.mean():.4f})",
                 fontsize=14, fontweight="bold")

    axes[0].bar(range(H), mse_per_step, color="#2196F3", alpha=0.9)
    for j, v in enumerate(mse_per_step):
        axes[0].text(j, v+0.001, f"{v:.3f}", ha="center", fontsize=7)
    axes[0].set_title("Per-Step MSE"); axes[0].grid(True, alpha=0.3, axis="y")

    if len(emb_traj) > 0:
        for d, col in zip(range(min(5, emb_traj.shape[1])),
                          ["#2196F3","#FF9800","#4CAF50","#F44336","#9C27B0"]):
            axes[1].plot(emb_traj[:, d], 'o-', color=col, ms=5, label=f"d{d}")
        axes[1].set_title("Embedding Trajectory (first 5 dims)")
        axes[1].legend(loc="upper right"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  T3 saved: {out} | mean MSE={mse_per_step.mean():.4f}")
    raw.cpu()


# ---------------------------------------------------------------------------
# T8: Embedding health
# ---------------------------------------------------------------------------

def test_t8_health(model, cfg, eval_idx, n_samples=500, out="t8_embedding_health.png"):
    from scipy import stats
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)

    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=4)
    embs = []
    with torch.no_grad():
        for i in range(min(n_samples, len(ds))):
            fr, _ = ds[i]
            e = raw.model.encode(fr.unsqueeze(0).to(dev)).cpu()
            embs.append(e[0, e.size(1)//2])

    E = torch.stack(embs).numpy()
    stds = E.std(0)
    collapse_thr = 0.01
    n_collapsed = (stds < collapse_thr).sum()

    # SVD for effective rank
    _, sv, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    sv = sv / sv.sum()
    eff_rank = np.exp(-(sv * np.log(sv + 1e-10)).sum())
    cum_var = np.cumsum(sv**2) / (sv**2).sum()
    d50 = np.searchsorted(cum_var, 0.5) + 1
    d90 = np.searchsorted(cum_var, 0.9) + 1

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"Embedding Health (H100, {n_samples} windows)", fontsize=13, fontweight="bold")

    # Per-dim std
    axes[0].bar(range(len(stds)), stds, color="#2196F3", alpha=0.8, width=1.0)
    axes[0].axhline(collapse_thr, color="red", ls="--", label="Collapse thr")
    axes[0].set_title(f"Per-Dim Std ({n_collapsed} collapsed / {len(stds)})")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Distribution vs Gaussian
    flat = E.flatten()
    axes[1].hist(flat, bins=80, density=True, color="#4CAF50", alpha=0.7, label="Embeddings")
    x = np.linspace(-4, 4, 200)
    axes[1].plot(x, stats.norm.pdf(x, 0, 0.1), 'r--', label="N(0.1)")
    axes[1].set_title("Distribution vs Gaussian — SIGReg target")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Singular values
    _, sv_raw, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    axes[2].plot(sv_raw / sv_raw.sum(), color="#2196F3")
    axes[2].set_title(f"Singular Values (Eff. rank={eff_rank:.1f}/{E.shape[1]})")
    axes[2].grid(True, alpha=0.3)

    # Cumulative variance
    axes[3].plot(cum_var, color="#4CAF50")
    axes[3].axhline(0.5, color="#FF9800", ls="--", label=f"50% @ {d50} dims")
    axes[3].axhline(0.9, color="red",     ls="--", label=f"90% @ {d90} dims")
    axes[3].set_title(f"Cumulative Variance (50%@{d50} dims, 90%@{d90} dims)")
    axes[3].legend(); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  T8 saved: {out}")
    print(f"     Collapsed dims: {n_collapsed}/{len(stds)}")
    print(f"     Effective rank: {eff_rank:.1f}/{E.shape[1]}")
    print(f"     50% variance @ {d50} dims, 90% @ {d90} dims")
    raw.cpu()
    return eff_rank


# ---------------------------------------------------------------------------
# T9: Temporal straightening
# ---------------------------------------------------------------------------

def test_t9_temporal(model, cfg, eval_idx, n_windows=300, out="t9_temporal.png"):
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)

    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=2)
    embs = []
    with torch.no_grad():
        for i in range(min(n_windows, len(ds))):
            fr, _ = ds[i]
            e = raw.model.encode(fr.unsqueeze(0).to(dev)).cpu()
            embs.append(e)

    E = torch.cat(embs, 0)          # (N, T, D)
    vels = E[:, 1:] - E[:, :-1]
    Tv = vels.shape[1]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(f"Temporal Straightening (H100, {n_windows} windows)",
                 fontsize=14, fontweight="bold")

    # Consecutive velocity cosine sim
    cpt = [F.cosine_similarity(vels[:,t], vels[:,t+1], dim=-1).mean().item()
           for t in range(Tv-1)] if Tv >= 2 else []
    if cpt:
        mean_cs = np.mean(cpt)
        axes[0].plot(cpt, 'o-', color="#FF9800", lw=2)
        axes[0].axhline(0, color="gray", ls="--")
        axes[0].axhline(mean_cs, color="red", ls=":", lw=2, label=f"mean={mean_cs:.4f}")
        axes[0].set_title("Consecutive velocity cosine sim")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        print(f"  Straightening mean: {mean_cs:.4f}")

    # Velocity magnitude boxplot
    vm = vels.norm(dim=-1)
    data = [vm[:,t].numpy() for t in range(Tv)]
    try:    axes[1].boxplot(data, tick_labels=[str(t) for t in range(Tv)])
    except: axes[1].boxplot(data, labels=[str(t) for t in range(Tv)])
    axes[1].set_title("Velocity Magnitudes per Step"); axes[1].grid(True, alpha=0.3)

    # PCA of 3 trajectory sequences
    for seq_i, col in [(0,"#2196F3"), (1,"#F44336"), (2,"#4CAF50")]:
        if seq_i >= len(E): continue
        traj = E[seq_i].numpy()
        u, sv, _ = np.linalg.svd(traj-traj.mean(0), full_matrices=False)
        proj = u[:,:2]*sv[:2]
        axes[2].plot(proj[:,0], proj[:,1], 'o-', color=col, ms=4, lw=1.5,
                      alpha=0.7, label=f"Seq {seq_i}")
    axes[2].set_title("3 Trajectory PCA Overlaid")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    # Velocity cosine similarity distribution
    all_cs = []
    for t in range(min(Tv-1, 5)):
        cs = F.cosine_similarity(vels[:,t], vels[:,t+1], dim=-1).numpy()
        all_cs.extend(cs.tolist())
    axes[3].hist(all_cs, bins=50, color="#9C27B0", alpha=0.8, density=True)
    axes[3].axvline(0, color='k', ls='--')
    if all_cs:
        axes[3].axvline(np.mean(all_cs), color='red', ls='-', lw=2,
                         label=f"mean={np.mean(all_cs):.3f}")
    axes[3].set_title("Velocity Cosine Sim Distribution")
    axes[3].set_xlabel("Cosine similarity"); axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  T9 saved: {out}")
    raw.cpu()


# ---------------------------------------------------------------------------
# TV1: Frame-to-frame cosine similarity heatmaps
# ---------------------------------------------------------------------------

def test_tv1_similarity(model, cfg, eval_idx, n_seqs=3, out="tv1_similarity.png"):
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)

    ds = Comma2k19Dataset(eval_idx, cfg, return_state=True, stride=8)
    fig, axes = plt.subplots(n_seqs, 4, figsize=(22, n_seqs*6))
    fig.suptitle("Frame-to-Frame Cosine Similarity in Latent Space",
                 fontsize=14, fontweight="bold")

    for seq_i in range(min(n_seqs, len(ds))):
        fr, _, st = ds[seq_i * max(len(ds)//n_seqs, 1)]
        with torch.no_grad():
            e = raw.model.encode(fr.unsqueeze(0).to(dev)).squeeze(0).cpu()
        en = F.normalize(e, dim=-1)
        sim_mat = (en @ en.T).numpy()
        speed = st[:,0].numpy()

        # Cosine sim matrix
        axes[seq_i,0].imshow(sim_mat, vmin=0.6, vmax=1.0, cmap="Greens")
        axes[seq_i,0].set_title(f"Seq {seq_i}: Cosine Sim Matrix")
        axes[seq_i,0].set_xlabel("Frame"); axes[seq_i,0].set_ylabel("Frame")

        # Drift from initial frame
        ax2 = axes[seq_i,1]
        drift = sim_mat[0]
        ax2.plot(drift, 'b-o', ms=4, label="Cosine sim")
        ax2_r = ax2.twinx()
        ax2_r.plot(speed, 'r--', alpha=0.7, label="Speed")
        ax2.set_title(f"Seq {seq_i}: Drift from initial frame")
        ax2.set_ylabel("Cosine sim"); ax2_r.set_ylabel("Speed (m/s)", color="r")
        ax2.grid(True, alpha=0.3)

        # Frame strip (thumbnail images)
        T = fr.shape[0]
        strip = _to_img(fr[::max(1, T//8)])
        strip_img = np.concatenate(strip, axis=1)
        axes[seq_i,2].imshow(strip_img)
        axes[seq_i,2].set_title(f"Seq {seq_i}: Frame Strip")
        axes[seq_i,2].axis("off")

        # Speed vs embedding norm
        norms = e.norm(dim=-1).numpy()
        axes[seq_i,3].plot(norms, 'b-', label="Emb norm")
        ax3_r = axes[seq_i,3].twinx()
        ax3_r.plot(speed, 'r--', alpha=0.7, label="Speed")
        axes[seq_i,3].set_title(f"Seq {seq_i}: Norm vs Speed")
        axes[seq_i,3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  TV1 saved: {out}")
    raw.cpu()


# ---------------------------------------------------------------------------
# VL1: Future selection accuracy
# ---------------------------------------------------------------------------

def test_vl1_future_selection(model, cfg, eval_idx, n_trials=8, n_distractors=3,
                               out="VL1_future_selection.png"):
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)

    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=10)
    if len(ds) < n_trials * (n_distractors + 1):
        print("  VL1: not enough data"); return

    correct = 0
    fig_rows = []

    for trial in range(n_trials):
        # Context = one sequence
        fr_ctx, ac_ctx = ds[trial * 5]
        # True future = next few frames from same sequence
        fr_true, _ = ds[trial * 5 + 1]

        # Distractors = random other sequences
        distractors = []
        for d in range(n_distractors):
            idx = (trial * 5 + 2 + d * 7) % len(ds)
            distractors.append(ds[idx][0])

        # Encode context
        with torch.no_grad():
            z_ctx = raw.model.encode(fr_ctx.unsqueeze(0).to(dev))
            z_ctx_last = z_ctx[0, -1]

            # Encode all candidates (true + distractors)
            candidates = [fr_true] + distractors
            z_cands = []
            for c in candidates:
                zc = raw.model.encode(c.unsqueeze(0).to(dev))[0, 0]
                z_cands.append(zc)

        # Find closest candidate in latent space
        dists = [((z_ctx_last - zc).pow(2).sum().item()) for zc in z_cands]
        model_pick = int(np.argmin(dists))
        if model_pick == 0: correct += 1

        fig_rows.append((fr_ctx, candidates, dists, model_pick))

    accuracy = correct / n_trials
    print(f"  VL1: {correct}/{n_trials} = {accuracy*100:.1f}% future selection accuracy")

    # Plot
    n_cols = 1 + (n_distractors + 1)  # context + candidates
    fig, axes = plt.subplots(n_trials, n_cols, figsize=(n_cols*3, n_trials*3.5))
    fig.suptitle("VL1: Can the model identify the correct future?\n"
                 "Col1=context | Col2-5=future choices | green=best match by model",
                 fontsize=11, fontweight="bold")

    for row, (fr_ctx, candidates, dists, pick) in enumerate(fig_rows):
        # Context strip
        ctx_imgs = _to_img(fr_ctx)
        strip = np.concatenate(ctx_imgs[::max(1, len(ctx_imgs)//3)][:3], axis=1)
        axes[row, 0].imshow(strip); axes[row, 0].axis("off")
        axes[row, 0].set_title("Context", color="#2196F3", fontsize=9)

        for c_i, (cand, dist) in enumerate(zip(candidates, dists)):
            img = _to_img(cand)[0]
            axes[row, c_i+1].imshow(img)
            axes[row, c_i+1].axis("off")
            color = "#4CAF50" if c_i == pick else "white"
            label = f"d={dist:.2f}"
            if c_i == pick:
                label = f"MODEL PICK\n{'TRUE' if pick==0 else 'WRONG'}\n{label}"
            axes[row, c_i+1].set_title(label, color=color, fontsize=8)
            for spine in axes[row, c_i+1].spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(2)

    plt.figtext(0.5, 0.01,
                f"Top-1 future selection accuracy: {correct}/{n_trials} = {accuracy*100:.1f}%",
                ha="center", fontsize=12,
                color="#4CAF50" if accuracy >= 0.75 else "#F44336",
                fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  VL1 saved: {out}")
    raw.cpu()


# ---------------------------------------------------------------------------
# BM1: Open-loop L2 error
# ---------------------------------------------------------------------------

def test_bm1_openloop(model, cfg, eval_idx, out="BM1_openloop.png"):
    """
    BM1: Compare model ADE to a null model (constant prediction = last known state).
    A good model should achieve ratio << 1.0 at all horizons.
    """
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)

    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=4)
    horizons = [1,2,4,6,8]
    model_ades = {h: [] for h in horizons}
    null_ades  = {h: [] for h in horizons}

    with torch.no_grad():
        for i in range(min(200, len(ds))):
            fr, ac = ds[i]
            emb = raw.model.encode(fr.unsqueeze(0).to(dev))
            ae  = raw.model.encode_actions(ac.unsqueeze(0).to(dev))
            pred = raw.model.predict(emb[:,:-1], ae[:,:-1])
            for h in horizons:
                if h < pred.shape[1]:
                    model_e = (pred[0, h-1] - emb[0, h]).pow(2).sum().sqrt().item()
                    null_e  = (emb[0, 0]   - emb[0, h]).pow(2).sum().sqrt().item()
                    model_ades[h].append(model_e)
                    null_ades[h].append(null_e)

    m_mean = {h: np.mean(v) for h, v in model_ades.items() if v}
    n_mean = {h: np.mean(v) for h, v in null_ades.items()  if v}
    ratios = {h: m_mean[h]/max(n_mean[h], 1e-6) for h in m_mean}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("BM1: Open-Loop L2 Error", fontsize=14, fontweight="bold")

    hs = sorted(m_mean.keys())
    axes[0].plot(hs, [m_mean[h] for h in hs], 'b-o', label="Model")
    axes[0].plot(hs, [n_mean[h] for h in hs], 'r--s', label="Null")
    axes[0].set_xlabel("Steps ahead"); axes[0].set_title("ADE vs Horizon")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].bar(hs, [ratios[h] for h in hs], color=["#4CAF50" if r < 0.6 else "#FF9800"
                                                      for r in [ratios[h] for h in hs]])
    axes[1].axhline(1.0, color="red", ls="--", label="Null")
    axes[1].axhline(0.6, color="green", ls=":", label="Good")
    for j, h in enumerate(hs):
        axes[1].text(j, ratios[h]+0.01, f"{ratios[h]:.2f}", ha="center", fontsize=9)
    axes[1].set_title("Model/Null Ratio"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    if 4 in model_ades and model_ades[4]:
        axes[2].hist(model_ades[4], bins=20, alpha=0.7, color="#2196F3", label="Model h=4")
        axes[2].hist(null_ades[4],  bins=20, alpha=0.7, color="#F44336", label="Null")
        axes[2].axvline(m_mean.get(4,0), color="#2196F3", ls="--")
        axes[2].axvline(n_mean.get(4,0), color="#F44336", ls="--")
    axes[2].set_title("Error dist h=4"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    mean_ratio = np.mean(list(ratios.values()))
    print(f"  BM1 saved: {out} | mean ratio={mean_ratio:.3f} "
          f"({'GOOD' if mean_ratio < 0.6 else 'NEEDS IMPROVEMENT'})")
    raw.cpu()


# ---------------------------------------------------------------------------
# BM2: Action sensitivity
# ---------------------------------------------------------------------------

def test_bm2_action_sensitivity(model, cfg, eval_idx, n_steps=20, out="BM2_action_sensitivity.png"):
    """
    BM2: Test whether different actions lead to diverging latent predictions.
    A responsive model should show non-trivial L2 divergence between action pairs.
    """
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)

    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=10)
    eA = cfg.effective_action_dim

    action_pairs = [
        ("Acc vs Brake",   torch.full((1,eA),-1.), torch.full((1,eA),1.)),
        ("Left vs Right",  torch.cat([-torch.ones(1,eA//3), torch.zeros(1,eA//3*2)],1),
                           torch.cat([torch.ones(1,eA//3),  torch.zeros(1,eA//3*2)],1)),
        ("Straight vs TurnL", torch.zeros(1,eA),
                               torch.cat([torch.ones(1,eA//3), torch.zeros(1,eA//3*2)],1)),
        ("Gentle vs HardL",   torch.cat([torch.full((1,eA//3),0.3), torch.zeros(1,eA//3*2)],1),
                               torch.cat([torch.full((1,eA//3),1.0), torch.zeros(1,eA//3*2)],1)),
        ("Coast vs Accel",    torch.zeros(1,eA), torch.full((1,eA),0.8)),
    ]

    results = {}
    with torch.no_grad():
        for name, a1, a2 in action_pairs:
            dists = []
            for i in range(min(50, len(ds))):
                fr, _ = ds[i]
                z0 = raw.model.encode(fr.unsqueeze(0).to(dev))[:,0:1]
                divs = []
                for step in range(n_steps):
                    ae1 = raw.model.encode_actions(a1.unsqueeze(0).to(dev))
                    ae2 = raw.model.encode_actions(a2.unsqueeze(0).to(dev))
                    p1  = raw.model.predict(z0, ae1)
                    p2  = raw.model.predict(z0, ae2)
                    divs.append((p1-p2).pow(2).mean().sqrt().item())
                    z0 = p1
                dists.append(divs)
            results[name] = np.array(dists).mean(0)

    mean_final = np.mean([v[-1] for v in results.values()])
    status = "GOOD" if mean_final > 0.05 else "WEAK"

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"BM2: Action Sensitivity | Mean={mean_final:.3f} | {status}",
                 fontsize=13, fontweight="bold")

    colors = ["#1565C0","#E65100","#1B5E20","#B71C1C","#6A1B9A"]
    for i, (name, vals) in enumerate(results.items()):
        ax = axes.flat[i]
        ax.fill_between(range(n_steps), vals*0.9, vals*1.1, alpha=0.3, color=colors[i%len(colors)])
        ax.plot(vals, 'o-', color=colors[i%len(colors)], lw=2, ms=4)
        ax.set_title(f"{name}\nfinal={vals[-1]:.3f}", color=colors[i%len(colors)])
        ax.set_xlabel("Step"); ax.set_ylabel("L2 div"); ax.grid(True, alpha=0.3)

    # Summary bar
    ax = axes.flat[5]
    names = list(results.keys())
    finals = [results[n][-1] for n in names]
    bars = ax.barh(names, finals, color=[colors[i%len(colors)] for i in range(len(names))])
    ax.axvline(mean_final, color="orange", ls="--", lw=2)
    ax.axvline(1.0, color="green", ls="--", lw=1)
    for j, (b, v) in enumerate(zip(bars, finals)):
        ax.text(v + 0.005, b.get_y() + b.get_height()/2, f"{v:.2f}", va="center", fontsize=9)
    ax.set_title("Summary"); ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  BM2 saved: {out} | mean={mean_final:.3f} ({status})")
    raw.cpu()


# ---------------------------------------------------------------------------
# BM3: CEM planner
# ---------------------------------------------------------------------------

def test_bm3_cem(model, cfg, eval_idx, n_ep=20, cem_iters=8,
                 cem_samples=50, cem_elites=8, horizon=4, out="BM3_cem.png"):
    """Evaluate CEM planning quality against random and zero-action baselines."""
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)

    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=10)
    eA = cfg.effective_action_dim

    def rollout(ctx_emb, acts):
        z = ctx_emb.clone()
        for t in range(acts.size(1)):
            a_step = acts[:, t:t+1].expand(z.size(0), z.size(1), -1)
            ae = raw.model.encode_actions(a_step)
            nx = raw.model.predict(z, ae)[:, -1:]
            z = torch.cat([z[:, 1:], nx], dim=1)
        return z[:, -1]

    d_cem, d_rand, d_zero = [], [], []

    with torch.no_grad():
        for ep in range(min(n_ep, len(ds))):
            fr, ac = ds[ep]
            emb = raw.model.encode(fr.unsqueeze(0).to(dev))
            ctx = emb[:, :cfg.history_size]
            goal = emb[:, min(cfg.seq_len - 1, cfg.history_size + horizon)]

            mu = torch.zeros(horizon, eA, device=dev)
            sig = torch.ones(horizon, eA, device=dev) * 0.7

            for _ in range(cem_iters):
                acts = (mu + sig * torch.randn(cem_samples, horizon, eA, device=dev)).clamp(-1, 1)
                goal_rep = goal.expand(cem_samples, -1)
                pred = rollout(ctx.expand(cem_samples, -1, -1), acts)
                dist = (pred - goal_rep).norm(dim=-1)
                _, elite_idx = dist.topk(cem_elites, largest=False)
                elites = acts[elite_idx]
                mu = elites.mean(0)
                sig = elites.std(0).clamp(min=0.05)

            cem_pred = rollout(ctx, mu.unsqueeze(0))[0]
            d_cem.append((cem_pred - goal[0]).norm().item())

            rand_acts = torch.empty(1, horizon, eA, device=dev).uniform_(-1, 1)
            rand_pred = rollout(ctx, rand_acts)[0]
            d_rand.append((rand_pred - goal[0]).norm().item())

            zero_acts = torch.zeros(1, horizon, eA, device=dev)
            zero_pred = rollout(ctx, zero_acts)[0]
            d_zero.append((zero_pred - goal[0]).norm().item())

    d_cem = np.array(d_cem)
    d_rand = np.array(d_rand)
    d_zero = np.array(d_zero)
    ratio = float(d_cem.mean() / max(d_rand.mean(), 1e-8))
    thresh = np.percentile(d_rand, 30)
    reach = float((d_cem < thresh).mean())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"BM3: CEM planner | ratio={ratio:.3f} | reach={reach:.1%}",
                 fontsize=13, fontweight="bold")

    for dist, label, col in [
        (d_cem, f"CEM {d_cem.mean():.3f}", "#2196F3"),
        (d_rand, f"Random {d_rand.mean():.3f}", "#F44336"),
        (d_zero, f"Zero {d_zero.mean():.3f}", "#4CAF50"),
    ]:
        axes[0].hist(dist, bins=18, alpha=0.65, label=label, color=col)
    axes[0].set_title("Goal distance distribution")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(d_rand, d_cem, c=np.arange(len(d_cem)), cmap="plasma", s=50, alpha=0.8)
    lim = max(d_rand.max(), d_cem.max()) * 1.1
    axes[1].plot([0, lim], [0, lim], "w--", lw=1.5)
    axes[1].set_xlim(0, lim); axes[1].set_ylim(0, lim)
    axes[1].set_xlabel("Random")
    axes[1].set_ylabel("CEM")
    axes[1].set_title("Per-episode CEM vs random")
    axes[1].grid(True, alpha=0.3)

    means = [d_cem.mean(), d_rand.mean(), d_zero.mean()]
    stds = [d_cem.std(), d_rand.std(), d_zero.std()]
    bars = axes[2].bar(["CEM", "Random", "Zero"], means, yerr=stds,
                       color=["#2196F3", "#F44336", "#4CAF50"], alpha=0.9)
    for b, v in zip(bars, means):
        axes[2].text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    axes[2].set_title("Mean goal distance")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  BM3 saved: {out} | ratio={ratio:.3f} reach={reach:.1%}")
    raw.cpu()
    return {"ratio": ratio, "reach_rate": reach}


# ---------------------------------------------------------------------------
# Additional tests requested for publication bundle
# ---------------------------------------------------------------------------

def test_tA_rollout(model, cfg, eval_idx, n_seqs=20, out="tA_rollout.png"):
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)
    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=10)

    all_mse = []
    with torch.no_grad():
        for i in range(min(n_seqs, len(ds))):
            fr, ac = ds[i]
            emb = raw.model.encode(fr.unsqueeze(0).to(dev))
            ae = raw.model.encode_actions(ac.unsqueeze(0).to(dev))
            pred = raw.model.predict(emb[:, :-1], ae[:, :-1])
            mse = ((pred[0] - emb[0, 1:]) ** 2).mean(dim=-1).cpu().numpy().tolist()
            all_mse.append(mse)

    max_len = max(len(m) for m in all_mse)
    arr = np.full((len(all_mse), max_len), np.nan)
    for i, m in enumerate(all_mse):
        arr[i, :len(m)] = m
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    x = np.arange(1, len(mean) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(x, mean, "o-", color="#2196F3", lw=2)
    axes[0].fill_between(x, mean - std, mean + std, color="#2196F3", alpha=0.25)
    axes[0].set_title("Rollout MSE vs horizon")
    axes[0].set_xlabel("Horizon")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(x, mean, "o-", color="#F44336", lw=2)
    axes[1].set_title("Rollout MSE (log scale)")
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("MSE (log)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("tA: Multi-horizon rollout stability", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  tA saved: {out}")
    raw.cpu()


def test_tB_retrieval(model, cfg, eval_idx, n_gallery=500, n_queries=100, out="tB_retrieval.png"):
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)
    ds = Comma2k19Dataset(eval_idx, cfg, return_state=True, stride=50)

    Z, S = [], []
    with torch.no_grad():
        for i in range(min(n_gallery, len(ds))):
            fr, _, st = ds[i]
            emb = raw.model.encode(fr.unsqueeze(0).to(dev))[0, cfg.seq_len // 2].cpu()
            Z.append(emb)
            S.append(st[cfg.seq_len // 2].cpu())

    Z = torch.stack(Z)
    S = torch.stack(S)
    Zn = F.normalize(Z, dim=-1)
    sim = Zn @ Zn.T
    sim.fill_diagonal_(-1e9)

    k = 5
    qn = min(n_queries, len(Z))
    speed_err, steer_err, rnd_speed, rnd_steer = [], [], [], []
    rng = np.random.RandomState(42)

    for qi in range(qn):
        _, kn = sim[qi].topk(k)
        nn_sp = S[kn, 0].mean().item()
        nn_st = S[kn, 1].abs().mean().item()
        gt_sp = S[qi, 0].item()
        gt_st = abs(S[qi, 1].item())
        speed_err.append(abs(gt_sp - nn_sp))
        steer_err.append(abs(gt_st - nn_st))

        ri = rng.choice(len(Z), size=k, replace=False)
        rnd_speed.append(abs(gt_sp - S[ri, 0].mean().item()))
        rnd_steer.append(abs(gt_st - S[ri, 1].abs().mean().item()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(rnd_speed, bins=20, alpha=0.6, color="gray", label=f"Random {np.mean(rnd_speed):.2f}")
    axes[0].hist(speed_err, bins=20, alpha=0.7, color="#2196F3", label=f"NN@{k} {np.mean(speed_err):.2f}")
    axes[0].set_title("Speed retrieval error")
    axes[0].set_xlabel("|query - retrieved mean| (m/s)")
    axes[0].legend(fontsize=8)

    axes[1].hist(rnd_steer, bins=20, alpha=0.6, color="gray", label=f"Random {np.mean(rnd_steer):.2f}")
    axes[1].hist(steer_err, bins=20, alpha=0.7, color="#FF9800", label=f"NN@{k} {np.mean(steer_err):.2f}")
    axes[1].set_title("Steering retrieval error")
    axes[1].set_xlabel("|query - retrieved mean| (deg)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  tB saved: {out}")
    raw.cpu()


def test_tB_retrieval_visual(model, cfg, eval_idx, out="tB_retrieval_visual.png"):
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)
    ds = Comma2k19Dataset(eval_idx, cfg, return_state=True, stride=20)
    if len(ds) < 6:
        print("  tB visual: not enough data")
        raw.cpu()
        return

    with torch.no_grad():
        ref_fr, _, _ = ds[0]
        ref_z = raw.model.encode(ref_fr.unsqueeze(0).to(dev))[0, cfg.seq_len // 2]
        cand = []
        for i in range(1, min(80, len(ds))):
            fr, _, _ = ds[i]
            z = raw.model.encode(fr.unsqueeze(0).to(dev))[0, cfg.seq_len // 2]
            d = (ref_z - z).pow(2).sum().sqrt().item()
            cand.append((d, fr))
        cand.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes[0, 0].imshow(_to_img(ref_fr[cfg.seq_len // 2].unsqueeze(0))[0])
    axes[0, 0].set_title("Query")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for j in range(3):
        d, fr = cand[j]
        axes[0, j + 1].imshow(_to_img(fr[cfg.seq_len // 2].unsqueeze(0))[0])
        axes[0, j + 1].set_title(f"Top-{j+1} d={d:.2f}")
        axes[0, j + 1].axis("off")
    for j in range(4):
        d, fr = cand[min(6 + j, len(cand) - 1)]
        axes[1, j].imshow(_to_img(fr[cfg.seq_len // 2].unsqueeze(0))[0])
        axes[1, j].set_title(f"Hard neg d={d:.2f}")
        axes[1, j].axis("off")

    fig.suptitle("tB retrieval visual")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  tB visual saved: {out}")
    raw.cpu()


def test_tdec1_interpolation(model, decoder, cfg, eval_idx, out="tdec1_interpolation.png"):
    if decoder is None:
        print("  tdec1: no decoder, skipping")
        return
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)
    decoder = decoder.to(dev); decoder.eval()
    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=30)
    if len(ds) < 2:
        print("  tdec1: not enough data")
        raw.cpu(); decoder.cpu(); return

    with torch.no_grad():
        z1 = raw.model.encode(ds[0][0].unsqueeze(0).to(dev))[0, cfg.seq_len // 2]
        z2 = raw.model.encode(ds[1][0].unsqueeze(0).to(dev))[0, cfg.seq_len // 2]
        alphas = torch.linspace(0, 1, 7, device=dev)
        imgs = []
        for a in alphas:
            z = (1 - a) * z1 + a * z2
            imgs.append(decoder(z.unsqueeze(0))[0].cpu().permute(1, 2, 0).numpy())

    fig, axes = plt.subplots(1, 7, figsize=(18, 3))
    for i, im in enumerate(imgs):
        axes[i].imshow(np.clip(im, 0, 1))
        axes[i].set_title(f"a={i/6:.2f}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  tdec1 saved: {out}")
    raw.cpu(); decoder.cpu()


def test_tdec2_heatmap(model, decoder, cfg, eval_idx, out="tdec2_heatmap.png"):
    if decoder is None:
        print("  tdec2: no decoder, skipping")
        return
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)
    decoder = decoder.to(dev); decoder.eval()
    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=25)

    fr, _ = ds[0]
    with torch.no_grad():
        real = _denorm(fr[cfg.seq_len // 2].unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
        z = raw.model.encode(fr.unsqueeze(0).to(dev))[0, cfg.seq_len // 2]
        rec = decoder(z.unsqueeze(0))[0].cpu().permute(1, 2, 0).numpy()
        err = np.abs(real - rec).mean(axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(np.clip(real, 0, 1)); axes[0].set_title("Real"); axes[0].axis("off")
    axes[1].imshow(np.clip(rec, 0, 1)); axes[1].set_title("Decoded"); axes[1].axis("off")
    im = axes[2].imshow(err, cmap="inferno")
    axes[2].set_title("Absolute error heatmap"); axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  tdec2 saved: {out}")
    raw.cpu(); decoder.cpu()


def test_tdec3_temporal(model, decoder, cfg, eval_idx, out="tdec3_temporal.png"):
    if decoder is None:
        print("  tdec3: no decoder, skipping")
        return
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)
    decoder = decoder.to(dev); decoder.eval()
    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=30)

    fr, _ = ds[0]
    with torch.no_grad():
        emb = raw.model.encode(fr.unsqueeze(0).to(dev))[0]
        dec = decoder(emb).cpu().permute(0, 2, 3, 1).numpy()
    diffs = np.mean(np.abs(dec[1:] - dec[:-1]), axis=(1, 2, 3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(diffs, "o-", color="#2196F3")
    axes[0].set_title("Frame-to-frame decoded delta")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Mean absolute delta")
    axes[0].grid(True, alpha=0.3)

    strip = np.concatenate([np.clip(dec[t], 0, 1) for t in np.linspace(0, len(dec)-1, 6, dtype=int)], axis=1)
    axes[1].imshow(strip)
    axes[1].set_title("Decoded temporal strip")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  tdec3 saved: {out}")
    raw.cpu(); decoder.cpu()


def test_tdec4_dim_perturb(model, decoder, cfg, eval_idx, out="tdec4_dim_perturb.png"):
    if decoder is None:
        print("  tdec4: no decoder, skipping")
        return
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)
    decoder = decoder.to(dev); decoder.eval()
    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=35)

    fr, _ = ds[0]
    with torch.no_grad():
        z = raw.model.encode(fr.unsqueeze(0).to(dev))[0, cfg.seq_len // 2]
        base = decoder(z.unsqueeze(0))[0].cpu().permute(1, 2, 0).numpy()
        idxs = [0, 1, 2, 8, 16, 32]
        ims = [base]
        for k in idxs:
            zp = z.clone()
            zp[k] += 2.0 * z.std().item()
            ims.append(decoder(zp.unsqueeze(0))[0].cpu().permute(1, 2, 0).numpy())

    fig, axes = plt.subplots(1, len(ims), figsize=(3 * len(ims), 3))
    titles = ["Base"] + [f"dim {k} +2sigma" for k in idxs]
    for i, im in enumerate(ims):
        axes[i].imshow(np.clip(im, 0, 1))
        axes[i].set_title(titles[i], fontsize=8)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  tdec4 saved: {out}")
    raw.cpu(); decoder.cpu()


def test_tsd1_speed_memory(model, cfg, eval_idx, out="tsd1_speed_memory.png"):
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)
    ds = Comma2k19Dataset(eval_idx, cfg, return_state=True, stride=12)

    lags, corrs = [], []
    with torch.no_grad():
        fr, _, st = ds[0]
        emb = raw.model.encode(fr.unsqueeze(0).to(dev))[0].cpu().numpy()
        spd = st[:, 0].numpy()
        for lag in range(1, min(10, len(spd) - 1)):
            v = np.linalg.norm(emb[lag:] - emb[:-lag], axis=-1)
            s = np.abs(spd[lag:] - spd[:-lag])
            r = np.corrcoef(v, s)[0, 1] if len(v) > 2 else 0.0
            lags.append(lag); corrs.append(r)

    plt.figure(figsize=(7, 4))
    plt.plot(lags, corrs, "o-", color="#2196F3")
    plt.axhline(0, color="gray", ls="--")
    plt.title("tsd1: speed-memory correlation by lag")
    plt.xlabel("Lag")
    plt.ylabel("corr(latent drift, speed delta)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  tsd1 speed-memory saved: {out}")
    raw.cpu()


def test_tsd3_speed(model, cfg, eval_idx, out="tsd3_speed.png"):
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)
    ds = Comma2k19Dataset(eval_idx, cfg, return_state=True, stride=8)

    fr, _, st = ds[0]
    with torch.no_grad():
        emb = raw.model.encode(fr.unsqueeze(0).to(dev))[0].cpu().numpy()
    spd = st[:, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(spd, "r-", lw=2)
    axes[0].set_title("Speed trace")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("m/s")
    axes[0].grid(True, alpha=0.3)

    pc = emb[:, 0]
    axes[1].scatter(spd, pc, c=np.arange(len(spd)), cmap="viridis", s=20)
    r = np.corrcoef(spd, pc)[0, 1] if len(spd) > 2 else 0.0
    axes[1].set_title(f"Speed vs latent dim0 (r={r:.3f})")
    axes[1].set_xlabel("Speed (m/s)")
    axes[1].set_ylabel("Latent dim0")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  tsd3 speed saved: {out}")
    raw.cpu()


def test_tsd4_planning(model, decoder, cfg, eval_idx, out="tsd4_planning.png"):
    if decoder is None:
        print("  tsd4: no decoder, skipping")
        return
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)
    decoder = decoder.to(dev); decoder.eval()
    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=20)
    eA = cfg.effective_action_dim

    fr, _ = ds[0]
    with torch.no_grad():
        emb = raw.model.encode(fr.unsqueeze(0).to(dev))
        ctx = emb[:, :cfg.history_size]
        goal = emb[:, min(cfg.seq_len - 1, cfg.history_size + 4)]

        mu = torch.zeros(4, eA, device=dev)
        sig = torch.ones(4, eA, device=dev) * 0.6

        def rollout(ctx_emb, acts):
            z = ctx_emb.clone()
            for t in range(acts.size(1)):
                a = acts[:, t:t+1].expand(z.size(0), z.size(1), -1)
                ae = raw.model.encode_actions(a)
                nx = raw.model.predict(z, ae)[:, -1:]
                z = torch.cat([z[:, 1:], nx], dim=1)
            return z[:, -1]

        for _ in range(8):
            acts = (mu + sig * torch.randn(60, 4, eA, device=dev)).clamp(-1, 1)
            pred = rollout(ctx.expand(60, -1, -1), acts)
            d = (pred - goal.expand(60, -1)).norm(dim=-1)
            _, idx = d.topk(10, largest=False)
            elites = acts[idx]
            mu = elites.mean(0)
            sig = elites.std(0).clamp(min=0.05)

        imag = []
        z = ctx.clone()
        for t in range(4):
            a = mu[t:t+1].unsqueeze(0).expand(1, z.size(1), -1)
            ae = raw.model.encode_actions(a)
            nx = raw.model.predict(z, ae)[:, -1:]
            imag.append(nx[0, 0])
            z = torch.cat([z[:, 1:], nx], dim=1)
        dec = decoder(torch.stack(imag)).cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(1, 6, figsize=(16, 3.2))
    axes[0].imshow(_to_img(fr[cfg.history_size - 1].unsqueeze(0))[0]); axes[0].set_title("Current")
    axes[0].axis("off")
    axes[1].imshow(np.clip(dec[0], 0, 1)); axes[1].set_title("Plan +1"); axes[1].axis("off")
    axes[2].imshow(np.clip(dec[1], 0, 1)); axes[2].set_title("Plan +2"); axes[2].axis("off")
    axes[3].imshow(np.clip(dec[2], 0, 1)); axes[3].set_title("Plan +3"); axes[3].axis("off")
    axes[4].imshow(np.clip(dec[3], 0, 1)); axes[4].set_title("Plan +4"); axes[4].axis("off")
    axes[5].imshow(_to_img(fr[min(cfg.seq_len - 1, cfg.history_size + 4)].unsqueeze(0))[0]); axes[5].set_title("Goal")
    axes[5].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  tsd4 planning saved: {out}")
    raw.cpu(); decoder.cpu()


def test_tE_summary(hist, out="tE_summary.png"):
    if not hist:
        print("  tE: no history, skipping")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("tE: training summary", fontsize=14, fontweight="bold")

    axes[0, 0].axis("off")
    metrics = [
        f"steps: {len(hist.get('pred', []))}",
        f"final pred: {hist.get('pred', [0])[-1]:.5f}",
        f"final sigreg: {hist.get('sigreg', [0])[-1]:.5f}",
        f"final total: {hist.get('total', [0])[-1]:.5f}",
    ]
    for i, txt in enumerate(metrics):
        axes[0, 0].text(0.05, 0.9 - i * 0.2, txt, fontsize=11)
    axes[0, 0].set_title("Run metrics")

    for ax, key, col in [
        (axes[0, 1], "pred", "#2196F3"),
        (axes[1, 0], "sigreg", "#F44336"),
        (axes[1, 1], "total", "#4CAF50"),
    ]:
        vals = hist.get(key, [])
        if vals:
            ax.plot(vals, color=col)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  tE saved: {out}")


# ---------------------------------------------------------------------------
# BM4: Decoder visual reconstruction
# ---------------------------------------------------------------------------

def test_bm4_visual(model, decoder, cfg, eval_idx, out="BM4_visual.png"):
    if decoder is None:
        print("  BM4: no decoder, skipping"); return
    dev = torch.device(cfg.device)
    raw = _unwrap(model); raw.eval(); raw.to(dev)
    decoder = decoder.to(dev); decoder.eval()

    ds = Comma2k19Dataset(eval_idx, cfg, return_state=False, stride=20)
    n_ctx = 3; n_future = 5; n_rows = 3

    fig, big_axes = plt.subplots(n_rows * 3 + n_rows, 8, figsize=(24, n_rows*12))
    fig.suptitle("BM4: Real vs Decoded | Orange border = IMAGINED future",
                 fontsize=13, fontweight="bold")

    MN = torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1)
    SD = torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)

    with torch.no_grad():
        for r in range(min(n_rows, len(ds))):
            fr, ac = ds[r * max(len(ds)//n_rows, 1)]
            fr_dev = fr.unsqueeze(0).to(dev)
            ac_dev = ac.unsqueeze(0).to(dev)

            emb = raw.model.encode(fr_dev)
            ae  = raw.model.encode_actions(ac_dev)
            pred = raw.model.predict(emb[:,:-1], ae[:,:-1])

            row_base = r * 3
            frames_real = (fr_dev * SD + MN).clamp(0,1)

            mses = []
            for t in range(n_ctx + n_future):
                col = t
                is_future = t >= n_ctx

                # Real
                real_img = frames_real[0, min(t, fr.shape[0]-1)].permute(1,2,0).cpu().numpy()
                big_axes[row_base, col].imshow(real_img)
                big_axes[row_base, col].axis("off")
                speed = fr.shape[0]  # placeholder
                label = f"+{t-n_ctx+1}" if is_future else "CTX"
                color = "#FF9800" if is_future else "#2196F3"
                big_axes[row_base, col].set_title(label, color=color, fontsize=8)

                # Decoded
                if t < n_ctx:
                    z = emb[0, t]
                    mse_val = 0.
                else:
                    z = pred[0, min(t-1, pred.shape[1]-1)]
                    real_z = emb[0, min(t, emb.shape[1]-1)]
                    mse_val = (z - real_z).pow(2).mean().item()
                    mses.append(mse_val)

                dec_img = decoder(z.unsqueeze(0)).squeeze(0).permute(1,2,0).cpu().numpy()
                big_axes[row_base+1, col].imshow(dec_img)
                big_axes[row_base+1, col].axis("off")
                if is_future:
                    big_axes[row_base+1, col].set_title(f"MSE={mse_val:.4f}", fontsize=7,
                                                          color="#FF9800")

            # MSE bar chart row
            ax_bar = big_axes[row_base+2, :]
            for ax in ax_bar: ax.axis("off")
            ax_bar[0].axis("on")
            mean_mse = np.mean(mses) if mses else 0
            ts_all = list(range(n_ctx + n_future))
            colors_bar = ["#2196F3"]*n_ctx + ["#FF9800"]*n_future
            ax_bar[0].bar(ts_all, [0]*n_ctx + mses, color=colors_bar, alpha=0.9, width=0.8)
            ax_bar[0].axhline(mean_mse, color="white", ls="--", label=f"mean={mean_mse:.4f}")
            ax_bar[0].legend(fontsize=8); ax_bar[0].set_xlim(-0.5, n_ctx+n_future-0.5)
            ax_bar[0].set_facecolor("#111111")

    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  BM4 saved: {out}")
    raw.cpu(); decoder.cpu()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(model, decoder, cfg, hist, eval_idx, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    def out(name): return os.path.join(output_dir, name)

    print(f"\n{'='*60}")
    print(f"  LeWM Self-Driving — Evaluation Suite")
    print(f"{'='*60}\n")

    tests = [
        ("T0: Training curves",        lambda: test_t0_curves(hist,                    out("t0_training_curves.png"))),
        ("T1: Data samples",           lambda: test_t1_data(cfg, eval_idx,             out("t1_data_samples.png"))),
        ("T2: Latent t-SNE",           lambda: test_t2_tsne(model, cfg, eval_idx,      out("t2_tsne.png"))),
        ("T3: Prediction quality",     lambda: test_t3_prediction(model, cfg, eval_idx,out("t3_prediction.png"))),
        ("T8: Embedding health",       lambda: test_t8_health(model, cfg, eval_idx,    out("t8_embedding_health.png"))),
        ("T9: Temporal straightening", lambda: test_t9_temporal(model, cfg, eval_idx,  out("t9_temporal.png"))),
        ("TV1: Similarity heatmaps",   lambda: test_tv1_similarity(model, cfg, eval_idx,out("tv1_similarity.png"))),
        ("VL1: Future selection",      lambda: test_vl1_future_selection(model, cfg, eval_idx, out("VL1_future_selection.png"))),
        ("BM1: Open-loop L2",          lambda: test_bm1_openloop(model, cfg, eval_idx, out("BM1_openloop.png"))),
        ("BM2: Action sensitivity",    lambda: test_bm2_action_sensitivity(model, cfg, eval_idx, out("BM2_action_sensitivity.png"))),
        ("BM3: CEM planner",           lambda: test_bm3_cem(model, cfg, eval_idx, out=out("BM3_cem.png"))),
        ("BM4: Visual decoder",        lambda: test_bm4_visual(model, decoder, cfg, eval_idx, out("BM4_visual.png"))),
        ("tA: Rollout",                lambda: test_tA_rollout(model, cfg, eval_idx, out=out("tA_rollout.png"))),
        ("tB: Retrieval",              lambda: test_tB_retrieval(model, cfg, eval_idx, out=out("tB_retrieval.png"))),
        ("tB: Retrieval visual",       lambda: test_tB_retrieval_visual(model, cfg, eval_idx, out=out("tB_retrieval_visual.png"))),
        ("tdec1: Interpolation",       lambda: test_tdec1_interpolation(model, decoder, cfg, eval_idx, out=out("tdec1_interpolation.png"))),
        ("tdec2: Heatmap",             lambda: test_tdec2_heatmap(model, decoder, cfg, eval_idx, out=out("tdec2_heatmap.png"))),
        ("tdec3: Temporal",            lambda: test_tdec3_temporal(model, decoder, cfg, eval_idx, out=out("tdec3_temporal.png"))),
        ("tdec4: Dim perturb",         lambda: test_tdec4_dim_perturb(model, decoder, cfg, eval_idx, out=out("tdec4_dim_perturb.png"))),
        ("tsd1: Speed memory",         lambda: test_tsd1_speed_memory(model, cfg, eval_idx, out=out("tsd1_speed_memory.png"))),
        ("tsd3: Speed",                lambda: test_tsd3_speed(model, cfg, eval_idx, out=out("tsd3_speed.png"))),
        ("tsd4: Planning",             lambda: test_tsd4_planning(model, decoder, cfg, eval_idx, out=out("tsd4_planning.png"))),
        ("tE: Summary",                lambda: test_tE_summary(hist, out=out("tE_summary.png"))),
    ]

    passed = failed = 0
    for name, fn in tests:
        print(f"  {name}")
        try:
            fn(); passed += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    FAILED: {e}"); failed += 1

    print(f"\n{'='*60}")
    print(f"  {passed} passed, {failed} failed")
    print(f"  Outputs written to: {output_dir}/")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="LeWM evaluation suite")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to final.pt checkpoint")
    p.add_argument("--data-dir",   type=str, default=None)
    p.add_argument("--lightning",  action="store_true")
    p.add_argument("--output-dir", type=str, default="./test_results")
    p.add_argument("--device",     type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    model, decoder, cfg, hist = load_checkpoint(args.checkpoint, args.device)

    if args.lightning:
        paths = lightning_ai_paths()
        cfg.processed_dir = paths["processed_dir"]
        if args.data_dir is None: cfg.data_dir = paths["data_dir"]
    if args.data_dir: cfg.data_dir = args.data_dir

    import numpy as _np
    real_idx = _np.load(
        os.path.join(cfg.processed_dir, "segment_index.npy"),
        allow_pickle=True).item()
    n = len(real_idx["segments"]); ne = max(1, n//10)
    eval_idx = {"segments": real_idx["segments"][-ne:]}

    run_all(model, decoder, cfg, hist, eval_idx, output_dir=args.output_dir)
