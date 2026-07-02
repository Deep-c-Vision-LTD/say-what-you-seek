"""
Score Gazeformer (CVPR'23, official checkpoint) on OUR COCO-Search18 val protocol,
as a density-map predictor, exactly mirroring how the ground-truth maps were built:

  - for each of our 326 (image, task) val pairs, sample N scanpaths with the
    official model (num_samples=10, its evaluation default);
  - map predicted fixations from Gazeformer's 512x320 space to the 1680x1050
    display space, aggregate ALL sampled fixations, blur with sigma=40 at display
    resolution (identical to scripts/process_cocosearch18.py's GT construction);
  - resize to 224 and score CC (+ NSS/AUC on the same rebuilt human fixation
    points used for xSoyelNet v3), full val and clean val (no train-image overlap).

Run from the gazeformer repo dir with its deps on PYTHONPATH:
  python eval_density_on_our_val.py
"""
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
TRAIN_JSON = "data/cocosearch18/proc/intents_train.json"
FIX_GLOB = "data/cocosearch18/coco_search18_fixations_TP_validation_split*.json"
OUT = "results/gazeformer_density_our_val_s70.json"

DISP_W, DISP_H = 1680, 1050
GF_W, GF_H = 512, 320
EVAL = 224
SIGMA_GT = 70
NUM_SAMPLES = 10
device = torch.device("cpu")   # official code is cuda-only; transformer runs fine on cpu

def cc_np(p, g):
    a, b = p - p.mean(), g - g.mean()
    d = np.sqrt((a*a).sum()*(b*b).sum()); return float((a*b).sum()/d) if d > 0 else 0.0

def load_map(path, size):
    m = np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), np.float64)
    m = m - m.min(); s = m.sum(); return m / s if s > 0 else m + 1.0 / m.size

def NSS(sal, fmap):
    s = (sal - sal.mean()) / (sal.std() + 1e-12)
    f = fmap > 0.5
    return float(s[f].mean()) if f.sum() else np.nan

def AUC_Judd(sal, fmap, n_thresh=100):
    s = sal.ravel(); s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    f = fmap.ravel() > 0.5
    if f.sum() == 0: return np.nan
    pos = s[f]; th = np.linspace(0, 1, n_thresh)
    tp = np.array([(pos >= t).mean() for t in th])
    fp = np.array([(s >= t).mean() for t in th])
    order = np.argsort(fp)
    return float(np.trapezoid(tp[order], fp[order]))

# --- model ---
transformer = Transformer(num_encoder_layers=6, nhead=8, d_model=512, num_decoder_layers=6,
                          dim_feedforward=512, img_hidden_dim=2048, lm_dmodel=768, device=device).to(device)
model = gazeformer(transformer=transformer, spatial_dim=(20, 32), max_len=7, device=device).to(device)
sd = torch.load(join(ASSETS, "checkpoints", "gazeformer_cocosearch_TP.pkg"), map_location=device)["model"]
model.load_state_dict(sd)
model.eval()
emb = np.load(join(ASSETS, "dataset", "embeddings.npy"), allow_pickle=True).item()
print("model + embeddings loaded", flush=True)

val = json.load(open(VAL_JSON))
train_imgs = {e["image"] for e in json.load(open(TRAIN_JSON))}
clean = [i for i, e in enumerate(val) if e["image"] not in train_imgs]

# human fixation points for NSS/AUC (same build as xSoyelNet v3 eval)
fix_pts = {}
for fp in sorted(glob.glob(FIX_GLOB)):
    for t in json.load(open(fp)):
        if t.get("condition") != "present" or t.get("correct") != 1: continue
        key = (t["name"], t["task"])
        fix_pts.setdefault(key, []).extend(
            zip([x*EVAL/DISP_W for x in t["X"]], [y*EVAL/DISP_H for y in t["Y"]]))

torch.manual_seed(42); np.random.seed(42)
ccs, nsss, aucs = [], [], []
for n, e in enumerate(val):
    base = e["image"].split("/")[-1]
    task = e["task"]
    fpath = join(FEAT_DIR, task.replace(" ", "_"), base.replace("jpg", "pth"))
    feats = torch.load(fpath, map_location="cpu").unsqueeze(0)
    scanpaths = run_model(model=model, src=feats, task=emb[task], device=device,
                          num_samples=NUM_SAMPLES)
    dens = np.zeros((DISP_H, DISP_W), np.float32)
    for sp in scanpaths:
        for y, x, t in sp:
            xi = int(np.clip(x * DISP_W / GF_W, 0, DISP_W - 1))
            yi = int(np.clip(y * DISP_H / GF_H, 0, DISP_H - 1))
            dens[yi, xi] += 1
    dens = gaussian_filter(dens, SIGMA_GT)
    pr = np.asarray(Image.fromarray(dens).resize((EVAL, EVAL), Image.BILINEAR), np.float64)
    gt = load_map(e["gt"], EVAL)
    ccs.append(cc_np(pr, gt))
    pts = fix_pts.get((base, task))
    if pts:
        fmap = np.zeros((EVAL, EVAL))
        for px, py in pts:
            fmap[int(min(max(py,0),EVAL-1)), int(min(max(px,0),EVAL-1))] = 1
        nsss.append(NSS(pr, fmap)); aucs.append(AUC_Judd(pr, fmap))
    else:
        nsss.append(np.nan); aucs.append(np.nan)
    if (n+1) % 50 == 0:
        print(f"  {n+1}/{len(val)}  running CC={np.mean(ccs):.4f}", flush=True)

ccs = np.array(ccs); nsss = np.array(nsss); aucs = np.array(aucs)
res = {"full_val_cc": float(np.nanmean(ccs)), "clean_val_cc": float(np.nanmean(ccs[clean])),
       "full_val_nss": float(np.nanmean(nsss)), "full_val_auc": float(np.nanmean(aucs)),
       "num_samples": NUM_SAMPLES, "sigma_gt": SIGMA_GT}
print(json.dumps(res, indent=2))
json.dump(res, open(OUT, "w"), indent=2)
print(f"saved -> {OUT}")
