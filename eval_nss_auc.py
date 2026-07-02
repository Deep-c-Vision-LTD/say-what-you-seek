"""
NSS/AUC for IntentNet v3 on COCO-Search18 val: rebuild per-(image,task) fixation
POINTS from the raw validation fixation jsons (display space 1680x1050, matching
scripts/process_cocosearch18.py conventions), scale to the 224x224 eval grid, and
score the v3 matched-intent predictions with NSS and AUC-Judd alongside CC/SIM.
"""
import glob, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from model import IntentASPP
from src.metrics import NSS, AUC_Judd

VAL_JSON = "data/cocosearch18/proc/intents_val.json"
FIX_GLOB = "data/cocosearch18/coco_search18_fixations_TP_validation_split*.json"
CKPT = "results/intent_v3.pt"
DISP_W, DISP_H = 1680, 1050
SIZE, EVAL, SIGMA = 448, 224, 1
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)

def cc_np(p, g):
    a, b = p - p.mean(), g - g.mean()
    d = np.sqrt((a*a).sum()*(b*b).sum()); return float((a*b).sum()/d) if d > 0 else 0.0
def sim_np(p, g):
    p = np.clip(p,0,None); p = p/(p.sum()+1e-9); g = g/(g.sum()+1e-9)
    return float(np.minimum(p, g).sum())
def load_map(path, size):
    m = np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), np.float64)
    m = m - m.min(); s = m.sum(); return m / s if s > 0 else m + 1.0 / m.size

# --- rebuild fixation point maps per (image basename, task), correct trials only ---
fix_pts = {}
for fp in sorted(glob.glob(FIX_GLOB)):
    for t in json.load(open(fp)):
        if t.get("condition") != "present" or t.get("correct") != 1:
            continue
        key = (t["name"], t["task"])
        xs = [x * EVAL / DISP_W for x in t["X"]]
        ys = [y * EVAL / DISP_H for y in t["Y"]]
        fix_pts.setdefault(key, []).extend(zip(xs, ys))
print(f"fixation groups: {len(fix_pts)}", flush=True)

val = json.load(open(VAL_JSON))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = IntentASPP().to(dev)
model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
model.eval()
intents = sorted({e["intent"] for e in val})
with torch.no_grad():
    temb = model.text_embed(intents, dev)
t_idx = {s: i for i, s in enumerate(intents)}

aggs = {"CC": [], "SIM": [], "NSS": [], "AUC": []}
n_missing = 0
with torch.no_grad():
    for e in val:
        base = e["image"].split("/")[-1]
        key = (base, e["task"])
        pts = fix_pts.get(key)
        im = Image.open(e["image"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        x = torch.from_numpy(((np.asarray(im, np.float32)/255 - IM_M)/IM_S).transpose(2,0,1))[None].to(dev)
        f = model.visual_features(x)
        s = model.readout_forward(f, temb[t_idx[e["intent"]]][None])
        pr = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear", align_corners=False))[0,0].cpu().numpy().astype(np.float64)
        pr = gaussian_filter(pr, SIGMA)
        gt = load_map(e["gt"], EVAL)
        aggs["CC"].append(cc_np(pr, gt)); aggs["SIM"].append(sim_np(pr, gt))
        if pts:
            fmap = np.zeros((EVAL, EVAL))
            for px, py in pts:
                xi, yi = int(min(max(px, 0), EVAL-1)), int(min(max(py, 0), EVAL-1))
                fmap[yi, xi] = 1
            aggs["NSS"].append(NSS(pr, fmap)); aggs["AUC"].append(AUC_Judd(pr, fmap))
        else:
            n_missing += 1

print(f"pairs without fixation match: {n_missing}/{len(val)}")
out = {k: [float(np.mean(v)), len(v)] for k, v in aggs.items()}
for k, (m, n) in out.items():
    print(f"  {k}: {m:.4f}  (n={n})")
json.dump(out, open("results/intent_v3_nss_auc.json","w"), indent=2)
print("saved -> checkpoints/intent_v3_nss_auc.json")
