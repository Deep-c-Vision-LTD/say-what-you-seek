"""
Conditioning ablations for IntentNet v3 (core claim re-verification):
  matched intent  vs  unconditional ("")  vs  mismatched intent (derangement of tasks).
Also: clean-val variant excluding the 28 images that also appear in train.
"""
import json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from model import IntentASPP

TRAIN_JSON = "data/cocosearch18/proc/intents_train.json"
VAL_JSON = "data/cocosearch18/proc/intents_val.json"
CKPT = "results/intent_v3.pt"
SIZE, EVAL = 448, 224
SIGMA = 1  # train-selected value from the training run
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)

def cc_np(p, g):
    a, b = p - p.mean(), g - g.mean()
    d = np.sqrt((a*a).sum()*(b*b).sum()); return float((a*b).sum()/d) if d > 0 else 0.0

def load_map(path, size):
    m = np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), np.float64)
    m = m - m.min(); s = m.sum(); return m / s if s > 0 else m + 1.0 / m.size

dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = IntentASPP().to(dev)
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd, strict=False)
model.eval()

train = json.load(open(TRAIN_JSON)); val = json.load(open(VAL_JSON))
train_imgs = {e["image"] for e in train}
clean_idx = [i for i, e in enumerate(val) if e["image"] not in train_imgs]
print(f"val={len(val)}  clean-val (no train-image overlap)={len(clean_idx)}", flush=True)

intents = sorted({e["intent"] for e in val})
with torch.no_grad():
    temb = model.text_embed(intents, dev)
    uncond = model.text_embed([""], dev)
t_idx = {s: i for i, s in enumerate(intents)}

# mismatched: rotate each entry's task to a different one (deterministic derangement)
def mismatch(intent):
    i = t_idx[intent]
    return intents[(i + 1) % len(intents)]

feats = {}
with torch.no_grad():
    for i, e in enumerate(val):
        if e["image"] in feats: continue
        im = Image.open(e["image"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        x = torch.from_numpy(((np.asarray(im, np.float32)/255 - IM_M)/IM_S).transpose(2,0,1))[None].to(dev)
        feats[e["image"]] = model.visual_features(x)[0].half().cpu()
        if (i+1) % 100 == 0: print(f"  cached {i+1}", flush=True)

@torch.no_grad()
def predict(e, cond):
    f = feats[e["image"]].float()[None].to(dev)
    s = model.readout_forward(f, cond)
    pr = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear", align_corners=False))[0,0].cpu().numpy().astype(np.float64)
    return gaussian_filter(pr, SIGMA) if SIGMA > 0 else pr

gts = [load_map(e["gt"], EVAL) for e in val]
results = {}
for name, cond_fn in [
    ("matched", lambda e: temb[t_idx[e["intent"]]][None]),
    ("unconditional", lambda e: uncond),
    ("mismatched", lambda e: temb[t_idx[mismatch(e["intent"])]][None]),
]:
    ccs = [cc_np(predict(e, cond_fn(e)), g) for e, g in zip(val, gts)]
    ccs = np.array(ccs)
    results[name] = {"full_val": float(ccs.mean()), "clean_val": float(ccs[clean_idx].mean())}
    print(f"{name:14s}: full val CC={ccs.mean():.4f}   clean val CC={ccs[clean_idx].mean():.4f}", flush=True)

json.dump(results, open("results/intent_v3_ablations.json","w"), indent=2)
print("saved -> checkpoints/intent_v3_ablations.json")
