"""Correct the (x,y)/(y,x) transpose in the HAT job's saved raw points and run the
full sigma steelman + NSS/AUC locally. Saved points were [row=clip(fx*H), col=clip(fy*W),
count]; recover x=row*W/H, y=col*H/W."""
import json, glob
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

DISP_W, DISP_H, EVAL = 1680, 1050, 224
raw = json.load(open("results/hat_raw_points.json"))
val = json.load(open("data/cocosearch18/proc/intents_val.json"))
train_imgs = {e["image"].split("/")[-1] for e in json.load(open("data/cocosearch18/proc/intents_train.json"))}

def cc_np(p, g):
    a, b = p - p.mean(), g - g.mean()
    d = np.sqrt((a*a).sum()*(b*b).sum()); return float((a*b).sum()/d) if d > 0 else 0.0
def load_map(path, size):
    m = np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), np.float64)
    m = m - m.min(); s = m.sum(); return m / s if s > 0 else m + 1.0 / m.size
def NSS(sal, fmap):
    s = (sal - sal.mean()) / (sal.std() + 1e-12); f = fmap > 0.5
    return float(s[f].mean()) if f.sum() else np.nan
def AUC_Judd(sal, fmap, n_thresh=100):
    s = sal.ravel(); s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    f = fmap.ravel() > 0.5
    if f.sum() == 0: return np.nan
    pos = s[f]; th = np.linspace(0, 1, n_thresh)
    tp = np.array([(pos >= t).mean() for t in th]); fp = np.array([(s >= t).mean() for t in th])
    o = np.argsort(fp); return float(np.trapezoid(tp[o], fp[o]))

fix_pts = {}
for fp in sorted(glob.glob("data/cocosearch18/coco_search18_fixations_TP_validation_split*.json")):
    for t in json.load(open(fp)):
        if t.get("condition") != "present" or t.get("correct") != 1: continue
        fix_pts.setdefault((t["name"], t["task"]), []).extend(
            zip([x*EVAL/DISP_W for x in t["X"]], [y*EVAL/DISP_H for y in t["Y"]]))

entries, gts, clean = [], [], []
for e in val:
    base = e["image"].split("/")[-1]
    key = base + "|" + e["task"]
    pts = raw[key]
    dens = np.zeros((EVAL, EVAL), np.float32)   # build at eval res; blur with per-axis
    for row, col, cnt in pts:                    # sigma scaled like the GT's resize
        fx = row / DISP_H; fy = col / DISP_W     # undo transpose (normalized coords)
        x = min(int(fx * EVAL), EVAL - 1); y = min(int(fy * EVAL), EVAL - 1)
        dens[y, x] += cnt
    entries.append((base, e["task"], dens))
    gts.append(load_map(e["gt"], EVAL))
    clean.append(base not in train_imgs)
clean = np.array(clean)

best = (None, -1)
for sig in [30, 40, 50, 60, 70, 80, 100]:
    ccs = []
    for (base, task, dens), gt in zip(entries, gts):
        pr = gaussian_filter(dens.astype(np.float64), (sig*EVAL/DISP_H, sig*EVAL/DISP_W))
        ccs.append(cc_np(pr, gt))
    m = float(np.mean(ccs))
    print(f"sigma={sig}: CC={m:.4f}")
    if m > best[1]: best = (sig, m)

sig = best[0]
ccs, nsss, aucs = [], [], []
for (base, task, dens), gt in zip(entries, gts):
    pr = gaussian_filter(dens.astype(np.float64), (sig*EVAL/DISP_H, sig*EVAL/DISP_W))
    ccs.append(cc_np(pr, gt))
    pts = fix_pts.get((base, task))
    if pts:
        fmap = np.zeros((EVAL, EVAL))
        for px, py in pts:
            fmap[int(min(max(py,0),EVAL-1)), int(min(max(px,0),EVAL-1))] = 1
        nsss.append(NSS(pr, fmap)); aucs.append(AUC_Judd(pr, fmap))
ccs = np.array(ccs)
res = {"best_sigma": sig, "full_val_cc": float(ccs.mean()), "clean_val_cc": float(ccs[clean].mean()),
       "nss": float(np.nanmean(nsss)), "auc": float(np.nanmean(aucs))}
print(json.dumps(res, indent=2))
json.dump(res, open("results/hat_density_our_val_corrected.json","w"), indent=2)
