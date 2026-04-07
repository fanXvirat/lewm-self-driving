#!/usr/bin/env python3
"""
LeWM Self-Driving — Comma2k19 World Model
==========================================
Implementation of LeWorldModel (arXiv:2603.19312) applied to self-driving
on the comma2k19 dataset. Trains a Joint-Embedding Predictive Architecture
(JEPA) end-to-end from raw pixels using only MSE prediction loss and SIGReg.

Usage:
  python lewm_v5_final.py --lightning --chunks 1 2 3 4 --budget-hours 2.9

Hardware: NVIDIA H100 80GB recommended. Falls back gracefully to smaller GPUs.
Dataset:  comma2k19 (Hugging Face: commaai/comma2k19)
"""

import os, sys, math, time, warnings, argparse, zipfile, shutil, subprocess
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from einops import rearrange
from multiprocessing import Pool, cpu_count

try:
    from torch.amp import autocast as _autocast, GradScaler
    AMP_NEW = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler
    AMP_NEW = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def notebook_install(quiet=True):
    pkgs = ["torch torchvision", "transformers einops",
            "matplotlib scipy scikit-learn", "av", "huggingface_hub"]
    for pkg in pkgs:
        cmd = [sys.executable, "-m", "pip", "install"] + pkg.split()
        if quiet: cmd += ["-q"]
        subprocess.run(cmd, check=False)
    print("packages installed")


def lightning_ai_paths(base="/teamspace/studios/this_studio"):
    storage = os.path.join(base, "storage")
    os.makedirs(storage, exist_ok=True)
    return {"data_dir":       os.path.join(storage, "comma2k19"),
            "processed_dir":  os.path.join(storage, "comma2k19_processed"),
            "checkpoint_dir": os.path.join(storage, "checkpoints_lewm_v5")}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_comma2k19_chunk(chunk_id=1, dest_dir="./comma2k19",
                              cache_dir="./comma2k19_raw"):
    filename   = f"Chunk_{chunk_id}.zip"
    hf_path    = f"raw_data/{filename}"
    direct_url = (f"https://huggingface.co/datasets/commaai/comma2k19"
                  f"/resolve/main/{hf_path}?download=true")
    chunk_out  = os.path.join(dest_dir, f"Chunk_{chunk_id}")

    if os.path.isdir(chunk_out) and len(os.listdir(chunk_out)) > 0:
        n = sum(1 for r,d,f in os.walk(chunk_out)
                if any(fn.endswith(('.hevc','.mp4')) for fn in f))
        print(f"  Chunk_{chunk_id} already extracted ({n} videos)")
        return chunk_out

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(dest_dir,  exist_ok=True)
    zip_path = os.path.join(cache_dir, filename)

    if not os.path.exists(zip_path):
        print(f"  Downloading {filename} (~8-10 GB)...")
        t0, _ok = time.time(), False
        try:
            from huggingface_hub import hf_hub_download
            dl = hf_hub_download(repo_id="commaai/comma2k19",
                                  filename=hf_path, repo_type="dataset",
                                  local_dir=cache_dir)
            exp = os.path.join(cache_dir, hf_path)
            for p in [exp, dl]:
                if p and os.path.exists(p) and p != zip_path:
                    shutil.move(p, zip_path); break
            _ok = os.path.exists(zip_path)
        except Exception as e:
            print(f"  hf_hub failed ({e}), wget fallback...")
        if not _ok:
            ret = os.system(f'wget -q --show-progress -c -O "{zip_path}" "{direct_url}"')
            _ok = ret==0 and os.path.exists(zip_path) and os.path.getsize(zip_path)>1e6
        if not _ok:
            ret = os.system(f'curl -L --progress-bar -C - -o "{zip_path}" "{direct_url}"')
            _ok = ret==0 and os.path.exists(zip_path) and os.path.getsize(zip_path)>1e6
        if not _ok:
            raise RuntimeError(f"Download failed.\nManual: wget -O {zip_path} \"{direct_url}\"")
        print(f"  {os.path.getsize(zip_path)/1e9:.1f}GB in {(time.time()-t0)/60:.1f}min")
    else:
        print(f"  cached {zip_path}")

    print(f"  Extracting to {dest_dir} ...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for i, m in enumerate(zf.namelist()):
            try: zf.extract(m, dest_dir)
            except Exception: pass
            if (i+1)%5000==0: print(f"    {i+1}/{len(zf.namelist())}...")
    print(f"  Extracted in {(time.time()-t0)/60:.1f}min")
    return chunk_out


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Hardware
    device: str          = "cuda"
    precision: str       = "bf16"
    seed: int            = 3072
    num_workers: int     = 8
    compile_model: bool  = True

    # Encoder (ViT-Small backbone)
    img_size: int             = 224
    patch_size: int           = 14
    encoder_hidden: int       = 384
    encoder_heads: int        = 6
    encoder_layers: int       = 12
    encoder_intermediate: int = 1536

    # Embedding
    embed_dim: int    = 384
    history_size: int = 8
    seq_len: int      = 20

    # Predictor
    pred_depth: int    = 8
    pred_heads: int    = 16
    pred_dim_head: int = 64
    pred_mlp_dim: int  = 2048
    pred_dropout: float = 0.1

    # Projectors
    proj_hidden: int = 2048

    # Actions
    action_dim: int             = 3
    frameskip: int              = 5
    action_embedder_smooth: int = 15

    # SIGReg
    sigreg_weight: float     = 0.025
    sigreg_warmup_steps: int = 2000
    sigreg_knots: int        = 17
    sigreg_num_proj: int     = 1024

    # Repr loss — cosine similarity for temporally close frames
    repr_loss_weight: float = 0.02

    # Training
    batch_size: int     = 128
    grad_accum: int     = 1
    num_epochs: int     = 30
    lr: float           = 1e-4
    weight_decay: float = 1e-3
    grad_clip: float    = 1.0
    warmup_epochs: int  = 2
    dataset_stride: int = 2

    # Paths
    data_dir: str       = "./comma2k19"
    processed_dir: str  = "./comma2k19_processed"
    checkpoint_dir: str = "./checkpoints_v5"

    # CEM planner
    cem_horizon: int    = 8
    cem_samples: int    = 300
    cem_elites: int     = 30
    cem_iterations: int = 30

    # Logging
    log_every: int        = 20
    save_every: int       = 2000
    straighten_every: int = 200
    num_state_dims: int   = 6
    train_budget_sec: float = 2 * 3600 + 40 * 60

    @property
    def effective_action_dim(self): return self.frameskip * self.action_dim


def auto_configure(cfg: Config) -> Config:
    if not torch.cuda.is_available():
        cfg.device="cpu"; cfg.precision="fp32"; cfg.batch_size=4
        cfg.grad_accum=1; cfg.num_workers=0; cfg.compile_model=False
        cfg.num_epochs=3; cfg.dataset_stride=5; cfg.seq_len=14
        return cfg

    name = torch.cuda.get_device_name()
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc   = torch.cuda.get_device_capability()
    print(f"GPU: {name} | VRAM: {vram:.1f}GB | CC: {cc[0]}.{cc[1]}")
    cfg.precision = "bf16"

    if "H100" in name or "H200" in name or vram > 70:
        cfg.batch_size=128; cfg.grad_accum=1; cfg.num_workers=16
        cfg.compile_model=True; cfg.seq_len=20; cfg.num_epochs=30
        cfg.dataset_stride=2
    elif vram > 50:
        cfg.batch_size=96; cfg.grad_accum=1; cfg.num_workers=8
        cfg.compile_model=False; cfg.seq_len=18; cfg.num_epochs=25
        cfg.dataset_stride=2
    elif vram > 30:
        cfg.batch_size=64; cfg.grad_accum=2; cfg.num_workers=8
        cfg.compile_model=False; cfg.seq_len=16; cfg.num_epochs=25
        cfg.dataset_stride=3
    else:
        cfg.batch_size=32; cfg.grad_accum=4; cfg.num_workers=4
        cfg.compile_model=False; cfg.seq_len=14; cfg.num_epochs=20
        cfg.dataset_stride=4

    print(f"  batch={cfg.batch_size*cfg.grad_accum}, seq={cfg.seq_len}, "
          f"prec={cfg.precision}, workers={cfg.num_workers}, "
          f"compile={cfg.compile_model}, epochs={cfg.num_epochs}")
    return cfg


def _unwrap(m): return getattr(m, "_orig_mod", m)


@contextmanager
def inference_mode(model):
    was = model.training; model.eval()
    try: yield
    finally:
        if was: model.train()


# ---------------------------------------------------------------------------
# 2. SIGReg — Sketched Isotropic Gaussian Regulariser
#
# Prevents representation collapse by matching latent embeddings to an
# isotropic Gaussian via the Epps-Pulley test on random 1D projections.
# Reference: Balestriero & LeCun, arXiv:2511.08544
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t   = torch.linspace(0.2, 4.0, knots)
        dt  = (4.0 - 0.2) / (knots - 1)
        win = torch.exp(-t.square() / 2.0)
        w   = torch.full((knots,), 2*dt); w[[0,-1]] = dt
        self.register_buffer("t",       t)
        self.register_buffer("phi",     win)
        self.register_buffer("weights", w * win)

    def forward(self, proj):   # proj: (T, B, D)
        D = proj.size(-1)
        A = torch.randn(D, self.num_proj, device=proj.device, dtype=torch.float32)
        A = A / A.norm(p=2, dim=0, keepdim=True)
        xp  = proj.float() @ A
        x_t = xp.unsqueeze(-1) * self.t
        err = ((x_t.cos().mean(-3) - self.phi).square()
               + x_t.sin().mean(-3).square())
        return ((err @ self.weights) * D).mean()


# ---------------------------------------------------------------------------
# 3. Architecture
# ---------------------------------------------------------------------------

def modulate(x, s, sc): return x * (1 + sc) + s


class SafeBN1d(nn.Module):
    def __init__(self, n):
        super().__init__(); self.bn = nn.BatchNorm1d(n)
    def forward(self, x):
        if x.size(0) <= 1 and self.training:
            return F.batch_norm(x, self.bn.running_mean, self.bn.running_var,
                                self.bn.weight, self.bn.bias,
                                False, self.bn.momentum, self.bn.eps)
        return self.bn(x)


class FF(nn.Module):
    def __init__(self, d, h, drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d,h), nn.GELU(),
            nn.Dropout(drop), nn.Linear(h,d), nn.Dropout(drop))
    def forward(self, x): return self.net(x)


class Attn(nn.Module):
    def __init__(self, d, heads=8, dh=64, drop=0.):
        super().__init__()
        inner = dh*heads; self.heads=heads; self.drop=drop
        self.norm = nn.LayerNorm(d)
        self.to_qkv = nn.Linear(d, inner*3, bias=False)
        self.out = (nn.Sequential(nn.Linear(inner,d),nn.Dropout(drop))
                    if not (heads==1 and dh==d) else nn.Identity())
    def forward(self, x, causal=True):
        x = self.norm(x); dp = self.drop if self.training else 0.
        q,k,v = (rearrange(t,"b n (h d)->b h n d",h=self.heads)
                 for t in self.to_qkv(x).chunk(3,-1))
        o = F.scaled_dot_product_attention(q,k,v,dropout_p=dp,is_causal=causal)
        return self.out(rearrange(o,"b h n d->b n (h d)"))


class CondBlock(nn.Module):
    """Transformer block with AdaLN conditioning for action injection."""
    def __init__(self, d, heads, dh, mlp, drop=0.):
        super().__init__()
        self.attn=Attn(d,heads,dh,drop); self.ff=FF(d,mlp,drop)
        self.n1=nn.LayerNorm(d,elementwise_affine=False,eps=1e-6)
        self.n2=nn.LayerNorm(d,elementwise_affine=False,eps=1e-6)
        self.ada=nn.Sequential(nn.SiLU(),nn.Linear(d,6*d))
        nn.init.constant_(self.ada[-1].weight,0)
        nn.init.constant_(self.ada[-1].bias,0)
    def forward(self, x, c):
        sa,sca,ga,sm,scm,gm = self.ada(c).chunk(6,-1)
        x = x + ga*self.attn(modulate(self.n1(x),sa,sca))
        x = x + gm*self.ff(modulate(self.n2(x),sm,scm))
        return x


class ActionEmbedder(nn.Module):
    def __init__(self, in_d, sm_d, emb_d, sc=4):
        super().__init__()
        self.smooth=nn.Conv1d(in_d,sm_d,1)
        self.proj=nn.Sequential(nn.Linear(sm_d,sc*emb_d),nn.SiLU(),
                                 nn.Linear(sc*emb_d,emb_d))
    def forward(self, x):
        x=x.float().permute(0,2,1); x=self.smooth(x).permute(0,2,1)
        return self.proj(x)


class ProjMLP(nn.Module):
    def __init__(self, in_d, hid, out=None):
        super().__init__(); out=out or in_d
        self.net=nn.Sequential(nn.Linear(in_d,hid),SafeBN1d(hid),
                                nn.GELU(),nn.Linear(hid,out))
    def forward(self, x): return self.net(x)


class CausalTF(nn.Module):
    def __init__(self, in_d, hid, out, depth, heads, dh, mlp, drop=0.):
        super().__init__()
        self.norm=nn.LayerNorm(hid)
        self.inp=nn.Linear(in_d,hid) if in_d!=hid else nn.Identity()
        self.cp =nn.Linear(in_d,hid) if in_d!=hid else nn.Identity()
        self.op =nn.Linear(hid,out)  if hid!=out  else nn.Identity()
        self.layers=nn.ModuleList([CondBlock(hid,heads,dh,mlp,drop)
                                   for _ in range(depth)])
    def forward(self, x, c):
        x=self.inp(x); c=self.cp(c)
        for b in self.layers: x=b(x,c)
        return self.op(self.norm(x))


class ARPredictor(nn.Module):
    """Autoregressive causal Transformer predictor over frame embeddings."""
    def __init__(self, *, num_frames, depth, heads, mlp_dim, input_dim,
                 hidden_dim, output_dim=None, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        self.pos =nn.Parameter(torch.randn(1,num_frames,input_dim)*0.02)
        self.drop=nn.Dropout(emb_dropout)
        self.tf  =CausalTF(input_dim,hidden_dim,output_dim or input_dim,
                            depth,heads,dim_head,mlp_dim,dropout)
    def forward(self, x, c):
        T=x.size(1)
        return self.tf(self.drop(x+self.pos[:,:T]),c)


# ---------------------------------------------------------------------------
# 4. JEPA World Model
# ---------------------------------------------------------------------------

class JEPA(nn.Module):
    def __init__(self, encoder, predictor, action_encoder,
                 projector=None, pred_proj=None):
        super().__init__()
        self.encoder=encoder; self.predictor=predictor
        self.action_encoder=action_encoder
        self.projector=projector or nn.Identity()
        self.pred_proj=pred_proj or nn.Identity()

    def encode(self, pixels):
        b,t = pixels.shape[:2]
        flat=rearrange(pixels,"b t c h w->(b t) c h w")
        try:    out=self.encoder(pixel_values=flat,interpolate_pos_encoding=True)
        except: out=self.encoder(pixel_values=flat)
        cls=out.last_hidden_state[:,0]
        return rearrange(self.projector(cls),"(b t) d->b t d",b=b)

    def encode_actions(self, a): return self.action_encoder(a)

    def predict(self, emb, act):
        p=self.predictor(emb,act); b=emb.size(0)
        return rearrange(self.pred_proj(rearrange(p,"b t d->(b t) d")),
                         "(b t) d->b t d",b=b)

    @torch.no_grad()
    def rollout(self, init_emb, action_seq, history_size=8):
        B,S,T = action_seq.shape[:3]; HS=history_size
        emb=rearrange(init_emb.unsqueeze(1).expand(B,S,-1,-1),
                      "b s h d->(b s) h d").clone()
        act=rearrange(action_seq,"b s t a->(b s) t a")
        for ts in range(T):
            ec=emb[:,-HS:]; nc=ec.size(1)
            if ts<nc:
                pad=torch.zeros(act.size(0),max(0,nc-(ts+1)),act.size(-1),
                                device=act.device,dtype=act.dtype)
                ac=torch.cat([pad,act[:,:ts+1]],1)[:,-nc:]
            else:
                ac=act[:,ts-nc+1:ts+1]
            ae=self.action_encoder(ac)
            emb=torch.cat([emb,self.predict(ec,ae)[:,-1:]],1)
        return rearrange(emb,"(b s) t d->b s t d",b=B,s=S)


class DrivingLeWM(nn.Module):
    """
    LeWM adapted for self-driving on Comma2k19.

    Architecture:
      - Encoder: ViT (384-dim hidden, 12 layers, patch=14) -> 384-dim latent
      - Predictor: Causal Transformer (8 layers, 16 heads) with AdaLN action cond.
      - Loss: MSE prediction + SIGReg anti-collapse + repr cosine smoothness
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg=cfg; self._gs=0
        from transformers import ViTModel, ViTConfig
        enc=ViTModel(ViTConfig(
            hidden_size=cfg.encoder_hidden, num_attention_heads=cfg.encoder_heads,
            num_hidden_layers=cfg.encoder_layers,
            intermediate_size=cfg.encoder_intermediate,
            image_size=cfg.img_size, patch_size=cfg.patch_size,
            num_channels=3), add_pooling_layer=False)

        pred=ARPredictor(num_frames=cfg.seq_len,
                          input_dim=cfg.embed_dim, hidden_dim=cfg.embed_dim,
                          output_dim=cfg.embed_dim, depth=cfg.pred_depth,
                          heads=cfg.pred_heads, dim_head=cfg.pred_dim_head,
                          mlp_dim=cfg.pred_mlp_dim, dropout=cfg.pred_dropout)

        ae  =ActionEmbedder(cfg.effective_action_dim,
                             cfg.action_embedder_smooth, cfg.embed_dim)
        proj=ProjMLP(cfg.encoder_hidden, cfg.proj_hidden, cfg.embed_dim)
        pp  =ProjMLP(cfg.embed_dim,      cfg.proj_hidden, cfg.embed_dim)

        self.model =JEPA(enc, pred, ae, proj, pp)
        self.sigreg=SIGReg(cfg.sigreg_knots, cfg.sigreg_num_proj)

    def compute_loss(self, pixels, actions):
        c=self.cfg; actions=torch.nan_to_num(actions,0.)

        emb=self.model.encode(pixels)
        ae =self.model.encode_actions(actions)

        pred     =self.model.predict(emb[:,:-1], ae[:,:-1])
        pred_loss=(pred - emb[:,1:]).pow(2).mean()

        sigreg_w =min(1.0,self._gs/c.sigreg_warmup_steps)*c.sigreg_weight
        sig_loss =self.sigreg(emb.transpose(0,1))

        repr_loss = torch.tensor(0., device=pixels.device)
        if c.repr_loss_weight > 0 and emb.size(1) >= 3:
            z  =F.normalize(emb, dim=-1)
            repr_loss = (1 - (z[:,:-1]*z[:,1:]).sum(-1).mean()) * 0.5
            if emb.size(1) >= 4:
                repr_loss = repr_loss + (1-(z[:,:-2]*z[:,2:]).sum(-1).mean())*0.3

        total = (pred_loss
                 + sigreg_w * sig_loss
                 + c.repr_loss_weight * repr_loss)

        return {"loss":        total,
                "pred_loss":   pred_loss.detach(),
                "sigreg_loss": sig_loss.detach(),
                "repr_loss":   repr_loss.detach(),
                "sigreg_w":    sigreg_w}

    @torch.no_grad()
    def plan_cem(self, cur_frames, goal_frame, prev_plan=None):
        """Cross-Entropy Method planning in latent space."""
        c=self.cfg; dev=next(self.parameters()).device; was=self.training
        self.eval()
        try:
            zc=self.model.encode(cur_frames); zg=self.model.encode(goal_frame)[:,-1]
            H,eA=c.cem_horizon,c.effective_action_dim
            mu =(prev_plan[:H] if prev_plan is not None and len(prev_plan)>0
                 else torch.zeros(H,eA,device=dev))
            sig=torch.ones(H,eA,device=dev)*(0.3 if prev_plan is not None else 0.5)
            for _ in range(c.cem_iterations):
                acts=(mu+sig*torch.randn(c.cem_samples,H,eA,device=dev)).clamp(-1,1)
                pr=self.model.rollout(zc,acts.unsqueeze(0),c.history_size)
                _,idx=(pr[0,:,-1]-zg).pow(2).sum(-1).topk(c.cem_elites,largest=False)
                el=acts[idx]; mu=el.mean(0); sig=el.std(0).clamp(1e-4)*0.95
            return mu
        finally:
            if was: self.train()


# ---------------------------------------------------------------------------
# 5. Data Pipeline
# ---------------------------------------------------------------------------

def _check_hevc_cuvid() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True, text=True, timeout=5)
        return "hevc_cuvid" in result.stdout
    except Exception:
        return False

_HEVC_CUVID_AVAILABLE: Optional[bool] = None


def _decode_video_gpu(path: str, img_size: int,
                       max_frames: int = 1200) -> Optional[np.ndarray]:
    """Decode HEVC video, preferring NVDEC GPU decode, falling back to PyAV."""
    global _HEVC_CUVID_AVAILABLE
    if _HEVC_CUVID_AVAILABLE is None:
        _HEVC_CUVID_AVAILABLE = _check_hevc_cuvid()
        tag = "NVDEC enabled" if _HEVC_CUVID_AVAILABLE else "CPU PyAV fallback"
        print(f"  Video decode: {tag}")

    if _HEVC_CUVID_AVAILABLE:
        return _decode_video_ffmpeg_gpu(path, img_size, max_frames)
    return _decode_video_av(path, img_size, max_frames)


def _decode_video_ffmpeg_gpu(path: str, img_size: int,
                               max_frames: int = 1200) -> Optional[np.ndarray]:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-c:v", "hevc_cuvid",
        "-i", path,
        "-vf", f"hwdownload,format=nv12,scale={img_size}:{img_size},format=rgb24",
        "-frames:v", str(max_frames),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0 or len(result.stdout) == 0:
            return _decode_video_av(path, img_size, max_frames)
        frame_bytes = img_size * img_size * 3
        n_frames = len(result.stdout) // frame_bytes
        if n_frames == 0:
            return _decode_video_av(path, img_size, max_frames)
        arr = np.frombuffer(result.stdout[:n_frames*frame_bytes], dtype=np.uint8)
        return arr.reshape(n_frames, img_size, img_size, 3)
    except Exception:
        return _decode_video_av(path, img_size, max_frames)


def _decode_video_av(path: str, img_size: int,
                      max_frames: int = 1200) -> Optional[np.ndarray]:
    try:
        import av
    except ImportError:
        raise RuntimeError("pip install av")
    frames = []
    try:
        container = av.open(path)
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONREF"
        for packet in container.demux(stream):
            for frame in packet.decode():
                if len(frames) >= max_frames: break
                img = frame.to_ndarray(format="rgb24")
                frames.append(_resize_np(img, img_size))
            if len(frames) >= max_frames: break
        container.close()
    except Exception:
        return None
    return np.stack(frames) if frames else None


def _resize_np(img, size):
    from PIL import Image
    return np.array(Image.fromarray(img).resize((size,size),Image.BILINEAR),
                    dtype=np.uint8)


def _read_numpy_log(path: str) -> Optional[np.ndarray]:
    """
    Read comma2k19 CAN/IMU log files.
    Timestamp files ('t') are float64; value files are float32.
    """
    if not os.path.exists(path): return None
    basename = os.path.basename(path)
    try:
        arr = np.load(path, allow_pickle=False)
        if arr.size > 0: return arr
    except Exception: pass
    primary = np.float64 if basename == 't' else np.float32
    fallback = np.float32 if basename == 't' else np.float64
    for dtype in [primary, fallback]:
        try:
            arr = np.fromfile(path, dtype=dtype)
            if arr.size > 0:
                return arr.astype(primary)
        except Exception: pass
    return None


def _find_comma2k19_segments(data_dir):
    segs = []
    for root, dirs, files in os.walk(data_dir):
        if any(f.endswith(('.hevc', '.mp4')) for f in files):
            segs.append(root)
    return sorted(segs)


def _build_states_for_segment(seg_processed, cfg):
    """
    Build states.npy and actions.npy for a preprocessed segment.
    Reads CAN/IMU data from the raw processed_log folder.
    """
    from functools import partial

    def _npy_load(path):
        if not os.path.exists(path): return None
        try:
            arr = np.load(path, allow_pickle=False)
            if arr.size > 0: return arr
        except Exception: pass
        return None

    def _safe_interp(frame_times, t, v, lo, hi, default=0.0):
        if t is None or v is None or len(t) < 2:
            return np.full(len(frame_times), default, dtype=np.float32)
        if v.ndim > 1: v = v[:, 0]
        t = t.astype(np.float64).flatten()
        v = v.astype(np.float32).flatten()
        n = min(len(t), len(v)); t, v = t[:n], v[:n]
        order = np.argsort(t); t, v = t[order], v[order]
        _, ui = np.unique(t, return_index=True); t, v = t[ui], v[ui]
        t = t - t[0]
        out = np.interp(frame_times, t, v).astype(np.float32)
        bad = ~np.isfinite(out) | (out < lo) | (out > hi)
        out[bad] = default
        return out

    fp = os.path.join(seg_processed, "frames.npy")
    if not os.path.exists(fp): return None, None

    n_all = np.load(fp, mmap_mode='r').shape[0]
    fs = cfg.frameskip

    # find raw segment path
    name = os.path.basename(seg_processed)
    parts = name.split("_")
    chunk_part = parts[0] + "_" + parts[1]
    rest = "_".join(parts[2:])
    last_under = rest.rfind("_")
    route = rest[:last_under]; seg_num = rest[last_under + 1:]
    raw_seg = os.path.join(cfg.data_dir, chunk_part, route, seg_num)

    if not os.path.isdir(raw_seg): return None, None
    log = os.path.join(raw_seg, "processed_log")
    if not os.path.isdir(log): return None, None

    t_sp = _npy_load(os.path.join(log, "CAN/speed/t"))
    v_sp = _npy_load(os.path.join(log, "CAN/speed/value"))
    t_st = _npy_load(os.path.join(log, "CAN/steering_angle/t"))
    v_st = _npy_load(os.path.join(log, "CAN/steering_angle/value"))
    t_ws = _npy_load(os.path.join(log, "CAN/wheel_speed/t"))
    v_ws = _npy_load(os.path.join(log, "CAN/wheel_speed/value"))
    t_ac = _npy_load(os.path.join(log, "IMU/accelerometer/t"))
    v_ac = _npy_load(os.path.join(log, "IMU/accelerometer/value"))
    t_gy = _npy_load(os.path.join(log, "IMU/gyro/t"))
    v_gy = _npy_load(os.path.join(log, "IMU/gyro/value"))

    ft = None
    for cand in [os.path.join(raw_seg, "global_pose", "frame_times"),
                  os.path.join(raw_seg, "frame_times")]:
        arr = _npy_load(cand)
        if arr is not None and len(arr) >= n_all // fs:
            ft = arr.astype(np.float64).flatten()
            if len(ft) >= n_all: ft = ft[::fs][:n_all]
            ft = ft - ft[0]; break
    if ft is None:
        ft = np.arange(n_all, dtype=np.float64) / (20.0 / fs)

    N = len(ft)

    def _shift_interp(t_arr, v_arr, lo, hi, col=0):
        if t_arr is None or v_arr is None: return np.zeros(N, np.float32)
        t2 = t_arr.flatten(); t2 = t2 - t2[0]
        v2 = v_arr[:, col] if v_arr.ndim > 1 else v_arr.flatten()
        return _safe_interp(ft, t2, v2.astype(np.float32), lo, hi)

    speed = _safe_interp(ft,
        t_sp.flatten()-t_sp.flatten()[0] if t_sp is not None else None,
        v_sp, 0, 60)
    if speed.std() < 0.01 and v_ws is not None and t_ws is not None:
        tw = t_ws.flatten(); tw = tw - tw[0]
        vw = v_ws.mean(axis=1) if v_ws.ndim > 1 else v_ws.flatten()
        speed = _safe_interp(ft, tw, vw.astype(np.float32), 0, 60)

    steer  = _shift_interp(t_st, v_st, -800, 800, col=0)
    accel  = _shift_interp(t_ac, v_ac, -30,   30, col=0)
    gyro_z = _shift_interp(t_gy, v_gy, -10,   10, col=2)

    actions = np.stack([np.clip(steer/500,-1,1),
                         np.clip(speed/30, 0,1),
                         np.zeros(N, np.float32)], axis=1).astype(np.float32)
    states  = np.stack([speed, steer, accel, gyro_z,
                         np.clip(speed/40,0,1),
                         np.clip(steer/500,-1,1)], axis=1).astype(np.float32)
    return actions, states


def _process_segment(args):
    seg_path, cfg_dict, proc_dir, img_size = args
    cfg_obj = Config()
    for k, v in cfg_dict.items():
        if hasattr(cfg_obj, k): setattr(cfg_obj, k, v)

    seg_name = (os.path.basename(os.path.dirname(seg_path)) + "_" +
                os.path.basename(seg_path))
    seg_out = os.path.join(proc_dir, seg_name)
    frames_path = os.path.join(seg_out, "frames.npy")

    if os.path.exists(frames_path): return seg_out, True

    video = None
    for fn in ["video.hevc", "video.mp4", "fcamera.hevc"]:
        vp = os.path.join(seg_path, fn)
        if os.path.exists(vp):
            video = _decode_video_gpu(vp, img_size)
            if video is not None: break

    if video is None or len(video) < cfg_obj.seq_len: return None, False

    os.makedirs(seg_out, exist_ok=True)
    step = cfg_obj.frameskip
    frames = video[::step]
    np.save(frames_path, frames)
    return seg_out, False


def preprocess_comma2k19(cfg: Config, force=False) -> Optional[Dict]:
    os.makedirs(cfg.processed_dir, exist_ok=True)
    idx_path = os.path.join(cfg.processed_dir, "segment_index.npy")

    if not force and os.path.exists(idx_path):
        idx = np.load(idx_path, allow_pickle=True).item()
        if idx.get("segments"):
            print(f"  Loaded {len(idx['segments'])} preprocessed segments")
            return idx

    raw_segs = _find_comma2k19_segments(cfg.data_dir)
    if not raw_segs:
        print(f"  No video segments found in {cfg.data_dir}")
        return None

    cfg_dict = vars(cfg)
    args = [(s, cfg_dict, cfg.processed_dir, cfg.img_size) for s in raw_segs]

    results = []
    nw = min(cfg.num_workers, cpu_count())
    if nw > 1:
        with Pool(nw) as pool:
            for r in pool.imap_unordered(_process_segment, args):
                results.append(r)
    else:
        for a in args: results.append(_process_segment(a))

    segments = sorted([r[0] for r in results if r[0] is not None])

    for seg in segments:
        sp = os.path.join(seg, "states.npy")
        if not os.path.exists(sp):
            actions, states = _build_states_for_segment(seg, cfg)
            if states is not None:
                np.save(os.path.join(seg, "states.npy"),  states)
                np.save(os.path.join(seg, "actions.npy"), actions)

    idx = {"segments": segments}
    np.save(idx_path, idx)
    print(f"  Preprocessed {len(segments)} segments")
    return idx


class Comma2k19Dataset(Dataset):
    MN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    SD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, idx, cfg, return_state=False, stride=None):
        self.cfg=cfg; self.return_state=return_state
        self.sl=cfg.seq_len; stride=stride or cfg.dataset_stride
        self.traj=[]
        for seg in idx["segments"]:
            fp=os.path.join(seg,"frames.npy")
            if not os.path.exists(fp): continue
            n=np.load(fp,mmap_mode='r').shape[0]
            if n<self.sl: continue
            ap=os.path.join(seg,"actions.npy")
            if not os.path.exists(ap): continue
            for s in range(0,n-self.sl+1,stride):
                self.traj.append((seg,s))
        print(f"  Dataset: {len(idx['segments'])} segs | "
              f"{len(self.traj)} windows (stride={stride})")

    def __len__(self): return len(self.traj)

    def __getitem__(self, i):
        seg,start=self.traj[i]; end=start+self.sl
        fr=np.load(os.path.join(seg,"frames.npy"),mmap_mode='r')[start:end].copy()
        ac=np.load(os.path.join(seg,"actions.npy"),mmap_mode='r')[start:end].copy()
        fr_f=(fr.astype(np.float32)/255.-self.MN)/self.SD
        fr_f=fr_f.transpose(0,3,1,2)
        fs=self.cfg.frameskip
        ac_e=np.repeat(ac[:,np.newaxis,:],fs,axis=1).reshape(self.sl,-1).astype(np.float32)
        out=(torch.from_numpy(fr_f),torch.from_numpy(ac_e))
        if self.return_state:
            sp=os.path.join(seg,"states.npy")
            st=np.load(sp,mmap_mode='r')[start:end].copy() if os.path.exists(sp) else np.zeros((self.sl,6),np.float32)
            return (*out,torch.from_numpy(st.astype(np.float32)))
        return out

    def get_speed_weights(self) -> np.ndarray:
        """Per-window sampling weights for stratified speed distribution."""
        speeds = np.zeros(len(self.traj), dtype=np.float32)
        for i,(seg,start) in enumerate(self.traj):
            sp=os.path.join(seg,"states.npy")
            if os.path.exists(sp):
                st=np.load(sp,mmap_mode='r')[start:start+self.sl:4]
                speeds[i]=st[:,0].mean() if len(st)>0 else 15.
            else: speeds[i]=15.
        bins=np.digitize(speeds,[10.,20.])
        counts=np.bincount(bins,minlength=3).astype(float)+1
        return (1./counts[bins]).astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Latent Decoder (visualisation only, not used during JEPA training)
# ---------------------------------------------------------------------------

class LatentDecoder(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(embed_dim,1024),nn.GELU(),
                               nn.Linear(1024,256*7*7),nn.GELU())
        self.deconv=nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.ConvTranspose2d(128,64, 4,2,1),nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4,2,1),nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4,2,1),nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 3,  4,2,1),nn.Sigmoid())
    def forward(self, z):
        return self.deconv(self.fc(z).reshape(-1,256,7,7))


def train_decoder(model, decoder, cfg, dataset=None, num_epochs=25):
    dev=torch.device(cfg.device)
    decoder.to(dev); raw=_unwrap(model); raw.eval()
    dl=DataLoader(dataset,batch_size=64,shuffle=True,
                  num_workers=min(cfg.num_workers,4),drop_last=True,
                  pin_memory=(cfg.device=="cuda"))
    opt=torch.optim.Adam(decoder.parameters(),lr=3e-4)
    MN=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1)
    SD=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
    print(f"  Training decoder ({num_epochs} epochs)...")
    for ep in range(num_epochs):
        tot=0; n=0
        for batch in dl:
            fr=batch[0].to(dev)
            with torch.no_grad(): emb=raw.model.encode(fr)
            t_idx=torch.randint(0,fr.size(1),(fr.size(0),))
            z=emb[torch.arange(fr.size(0)),t_idx]
            tgt=fr[torch.arange(fr.size(0)),t_idx]
            tgt01=tgt*SD+MN; recon=decoder(z)
            mse=F.mse_loss(recon,tgt01)
            dx_r=recon[:,:,1:,:]-recon[:,:,:-1,:]
            dx_t=tgt01[:,:,1:,:]-tgt01[:,:,:-1,:]
            loss=mse+F.mse_loss(dx_r,dx_t)*0.1
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=mse.item(); n+=1
        if (ep+1)%5==0:
            print(f"    Decoder ep {ep+1}/{num_epochs}: MSE={tot/max(n,1):.5f}")
    decoder.eval(); return decoder


# ---------------------------------------------------------------------------
# 7. Training
# ---------------------------------------------------------------------------

def train(model: DrivingLeWM, cfg: Config, dataset: Dataset) -> Dict:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    dev=torch.device(cfg.device); model.to(dev)
    raw_model=model

    if cfg.compile_model and hasattr(torch,"compile"):
        print("  torch.compile...")
        try:
            model=torch.compile(model,mode="reduce-overhead")
            print("  compiled")
        except Exception as e:
            print(f"  compile failed ({e})")

    dl=DataLoader(dataset,batch_size=cfg.batch_size,shuffle=True,
                  num_workers=cfg.num_workers,pin_memory=(cfg.device=="cuda"),
                  drop_last=True,
                  **({"prefetch_factor":4,"persistent_workers":True}
                     if cfg.num_workers>0 else {}))

    try:
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,
                               weight_decay=cfg.weight_decay,fused=True)
    except (TypeError,RuntimeError):
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,
                               weight_decay=cfg.weight_decay)

    ospe=max(len(dl)//cfg.grad_accum,1)
    total_opt=cfg.num_epochs*ospe; warm=cfg.warmup_epochs*ospe

    def lr_fn(s):
        if s<warm: return s/max(warm,1)
        return 0.5*(1+math.cos(math.pi*(s-warm)/max(total_opt-warm,1)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_fn)

    use_amp=(cfg.precision in ("fp16","bf16") and dev.type=="cuda")
    if use_amp:
        amp_dt=(torch.bfloat16 if cfg.precision=="bf16"
                and torch.cuda.is_bf16_supported() else torch.float16)
        scaler=((GradScaler("cuda") if AMP_NEW else GradScaler())
                if amp_dt==torch.float16 else None)
    else:
        amp_dt=torch.float32; scaler=None

    total_steps=cfg.num_epochs*len(dl)
    print(f"\n{'='*60}")
    print(f"  LeWM Self-Driving — Training")
    print(f"{'='*60}")
    print(f"  Windows   : {len(dataset)}")
    print(f"  Effective batch : {cfg.batch_size*cfg.grad_accum}")
    print(f"  Seq length: {cfg.seq_len} -> {cfg.seq_len-1} targets/item")
    print(f"  Epochs    : {cfg.num_epochs} | ~{total_steps} steps")
    print(f"  Precision : {cfg.precision} | Compile: {cfg.compile_model}")
    print(f"  Budget    : {cfg.train_budget_sec/3600:.2f}h")
    print(f"{'='*60}\n")

    hist={"pred":[],"sigreg":[],"repr":[],"total":[],"lr":[],
          "sigreg_w":[],"straighten":[]}
    gs=0; t0=t_t0=time.time()
    model.train(); opt.zero_grad()

    for ep in range(cfg.num_epochs):
        if time.time()-t_t0>=cfg.train_budget_sec:
            print(f"\n  Budget reached, stopping at epoch {ep+1}"); break
        ep_p=ep_s=nb=0

        for bi,batch in enumerate(dl):
            if time.time()-t_t0>=cfg.train_budget_sec: break
            fr=batch[0].to(dev,non_blocking=True)
            ac=batch[1].to(dev,non_blocking=True)
            raw_model._gs=gs

            if use_amp:
                ctx=(_autocast(device_type="cuda",dtype=amp_dt) if AMP_NEW
                     else _autocast())
                with ctx: losses=model.compute_loss(fr,ac)
            else:
                losses=model.compute_loss(fr,ac)

            loss=losses["loss"]/cfg.grad_accum
            if scaler: scaler.scale(loss).backward()
            else:       loss.backward()

            if (bi+1)%cfg.grad_accum==0:
                if scaler: scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
                if scaler: scaler.step(opt); scaler.update()
                else:       opt.step()
                opt.zero_grad(); sched.step()

            gs+=1
            pl=losses["pred_loss"].item(); sl=losses["sigreg_loss"].item()
            rl=losses["repr_loss"].item(); sw=losses.get("sigreg_w",cfg.sigreg_weight)
            ep_p+=pl; ep_s+=sl; nb+=1
            hist["pred"].append(pl); hist["sigreg"].append(sl)
            hist["repr"].append(rl); hist["total"].append(losses["loss"].item())
            hist["lr"].append(sched.get_last_lr()[0]); hist["sigreg_w"].append(sw)

            if gs%cfg.log_every==0:
                el=time.time()-t0; eta=el/gs*(total_steps-gs)
                vr=torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
                thr=gs*cfg.batch_size/el
                print(f"[E{ep+1}/{cfg.num_epochs}] S{gs} | "
                      f"pred={pl:.4f} sig={sl:.4f}(x{sw:.3f}) "
                      f"repr={rl:.4f} | lr={sched.get_last_lr()[0]:.2e} | "
                      f"VRAM={vr:.1f}G | {thr:.0f} samp/s | ETA={eta/60:.0f}m")

            if gs%cfg.straighten_every==0:
                try:
                    with torch.no_grad():
                        emb=raw_model.model.encode(fr[:8])
                        vels=emb[:,1:]-emb[:,:-1]
                        if vels.shape[1]>=2:
                            v1=vels[:,:-1].reshape(-1,vels.shape[-1])
                            v2=vels[:,1:].reshape(-1,vels.shape[-1])
                            cs=F.cosine_similarity(v1,v2,dim=-1).mean().item()
                            hist["straighten"].append((gs,cs))
                except Exception: pass

            if gs%cfg.save_every==0:
                fp=os.path.join(cfg.checkpoint_dir,f"step_{gs}.pt")
                torch.save({"model": _unwrap(model).state_dict(),
                            "config": vars(cfg), "history": hist,
                            "step": gs}, fp)

        print(f"\n  Epoch {ep+1}: pred={ep_p/max(nb,1):.4f} "
              f"sig={ep_s/max(nb,1):.4f}")

    return hist


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def _parse_args():
    p=argparse.ArgumentParser(description="LeWM Self-Driving on Comma2k19")
    p.add_argument("--chunks",type=int,nargs="+",default=[1,2,3,4])
    p.add_argument("--install-only",action="store_true")
    p.add_argument("--data-dir",type=str,default=None)
    p.add_argument("--lightning",action="store_true")
    p.add_argument("--budget-hours",type=float,default=2.667)
    p.add_argument("--force-preprocess",action="store_true")
    p.add_argument("--skip-download",action="store_true")
    return p.parse_args()


def main():
    args=_parse_args()
    if args.install_only: notebook_install(quiet=False); return

    print(f"{'='*60}")
    print(f"  LeWM Self-Driving — Comma2k19")
    print(f"  Paper: arxiv.org/abs/2603.19312")
    print(f"{'='*60}\n")

    cfg=Config(); torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    cfg=auto_configure(cfg); cfg.train_budget_sec=args.budget_hours*3600

    if args.lightning:
        paths=lightning_ai_paths()
        cfg.data_dir=paths["data_dir"]; cfg.processed_dir=paths["processed_dir"]
        cfg.checkpoint_dir=paths["checkpoint_dir"]

    if args.data_dir: cfg.data_dir=args.data_dir

    if not args.skip_download:
        existing=_find_comma2k19_segments(cfg.data_dir) if os.path.isdir(cfg.data_dir) else []
        if not existing:
            for cid in args.chunks:
                print(f"\nDownloading Chunk_{cid}...")
                download_comma2k19_chunk(cid,dest_dir=cfg.data_dir)
        else:
            print(f"  Using existing data ({len(existing)} segs)")

    print("\nPreprocessing...")
    t_pp=time.time()
    real_idx=preprocess_comma2k19(cfg,force=args.force_preprocess)
    if real_idx is None or len(real_idx["segments"])==0:
        raise RuntimeError(f"No valid segments in {cfg.data_dir}")
    pp_t=time.time()-t_pp
    print(f"  Preprocessing took {pp_t/60:.1f}min")

    cfg.train_budget_sec=max(cfg.train_budget_sec-pp_t, 30*60)

    n=len(real_idx["segments"]); ne=max(1,n//10)
    train_idx={"segments":real_idx["segments"][:-ne]}
    eval_idx ={"segments":real_idx["segments"][-ne:]}

    train_ds=Comma2k19Dataset(train_idx,cfg)
    eval_ds =Comma2k19Dataset(eval_idx, cfg,return_state=True)
    print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    if len(train_ds)==0:
        raise RuntimeError("Training dataset empty after preprocessing.")

    print("\nBuilding model...")
    model=DrivingLeWM(cfg)
    np_=sum(p.numel() for p in model.parameters())
    print(f"  {np_:,} params ({np_/1e6:.1f}M)")

    dev=torch.device(cfg.device); model.to(dev)
    item=train_ds[0]
    tf=item[0].unsqueeze(0).to(dev); ta=item[1].unsqueeze(0).to(dev)
    with torch.no_grad():
        tl=model.compute_loss(tf,ta)
    print(f"  Sanity check: loss={tl['loss'].item():.4f} "
          f"pred={tl['pred_loss'].item():.4f} "
          f"sig={tl['sigreg_loss'].item():.4f}")
    model.cpu()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    hist=train(model,cfg,dataset=train_ds)

    print(f"\n{'='*60}")
    print(f"  Training decoder (25 epochs)")
    print(f"{'='*60}")
    decoder=LatentDecoder(cfg.embed_dim)
    decoder=train_decoder(_unwrap(model),decoder,cfg,
                           dataset=eval_ds,num_epochs=25)

    fp=os.path.join(cfg.checkpoint_dir,"final.pt")
    torch.save({"model":        _unwrap(model).state_dict(),
                "decoder":      decoder.state_dict(),
                "config":       vars(cfg),
                "history":      hist}, fp)
    print(f"\n  Saved: {fp}")
    print(f"\n{'='*60}")
    print(f"  Training complete. Run tests.py to evaluate.")
    print(f"{'='*60}")


if __name__=="__main__":
    main()
