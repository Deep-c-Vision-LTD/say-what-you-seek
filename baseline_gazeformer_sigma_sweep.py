"""Steelman check: sample 30 scanpaths per pair ONCE, then score Gazeformer's density
under {10,30} samples x sigma {20,40,60} -- report its BEST config, so the paper's
comparison gives the baseline every advantage."""
import json, glob
from os.path import join
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from models import Transformer
from gazeformer import gazeformer
from test import run_model

ASSETS = "gazeformer_assets"
FEAT_DIR = join(ASSETS, "dataset", "image_features")
VAL_JSON = "data/cocosearch18/proc/intents_val.json"
DISP_W, DISP_H, GF_W, GF_H, EVAL = 1680, 1050, 512, 320, 224
device = torch.device("cpu")

def cc_np(p, g):
    a, b = p - p.mean(), g - g.mean()
    d = np.sqrt((a*a).sum()*(b*b).sum()); return float((a*b).sum()/d) if d > 0 else 0.0
def load_map(path, size):
    m = np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), np.float64)
    m = m - m.min(); s = m.sum(); return m / s if s > 0 else m + 1.0 / m.size

transformer = Transformer(num_encoder_layers=6, nhead=8, d_model=512, num_decoder_layers=6,
                          dim_feedforward=512, img_hidden_dim=2048, lm_dmodel=768, device=device).to(device)
model = gazeformer(transformer=transformer, spatial_dim=(20, 32), max_len=7, device=device).to(device)
model.load_state_dict(torch.load(join(ASSETS, "checkpoints", "gazeformer_cocosearch_TP.pkg"), map_location=device)["model"])
model.eval()
emb = np.load(join(ASSETS, "dataset", "embeddings.npy"), allow_pickle=True).item()

val = json.load(open(VAL_JSON))
torch.manual_seed(42); np.random.seed(42)
all_pts, gts = [], []
for n, e in enumerate(val):
    base = e["image"].split("/")[-1]
    feats = torch.load(join(FEAT_DIR, e["task"].replace(" ", "_"), base.replace("jpg", "pth")), map_location="cpu").unsqueeze(0)
    sps = run_model(model=model, src=feats, task=emb[e["task"]], device=device, num_samples=30)
    pts = []
    for sp in sps:
        pts.append([(np.clip(x*DISP_W/GF_W, 0, DISP_W-1), np.clip(y*DISP_H/GF_H, 0, DISP_H-1)) for y, x, t in sp])
    all_pts.append(pts); gts.append(load_map(e["gt"], EVAL))
    if (n+1) % 100 == 0: print(f"  sampled {n+1}/{len(val)}", flush=True)

for n_s in [10]:
    for sig in [40, 50, 60, 70, 80, 100]:
        ccs = []
        for pts, gt in zip(all_pts, gts):
            dens = np.zeros((DISP_H, DISP_W), np.float32)
            for sp in pts[:n_s]:
                for x, y in sp:
                    dens[int(y), int(x)] += 1
            dens = gaussian_filter(dens, sig)
            pr = np.asarray(Image.fromarray(dens).resize((EVAL, EVAL), Image.BILINEAR), np.float64)
            ccs.append(cc_np(pr, gt))
        print(f"samples={n_s} sigma={sig}: CC={np.mean(ccs):.4f}", flush=True)
