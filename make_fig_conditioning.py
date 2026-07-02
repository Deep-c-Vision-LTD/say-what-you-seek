"""Fig 1: same image, two goals. Columns: Original | GT | Gazeformer | HAT | ours.
All from clean-val (image unseen in our training). Baselines at their steelman sigmas."""
import json, sys
import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from model import IntentASPP

IMG_BASE = "000000026564.jpg"
TASKS = ["bottle", "laptop"]
DISP_W, DISP_H, EVAL = 1680, 1050, 224
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)
val = json.load(open("data/cocosearch18/proc/intents_val.json"))
entries = {e["task"]: e for e in val if e["image"].endswith(IMG_BASE) and e["task"] in TASKS}

dev = "mps" if torch.backends.mps.is_available() else "cpu"

# --- ours ---
ours = IntentASPP().to(dev)
ours.load_state_dict(torch.load("results/intent_v3.pt", map_location="cpu"), strict=False)
ours.eval()
def ours_pred(img_path, intent):
    im = Image.open(img_path).convert("RGB").resize((448, 448), Image.BILINEAR)
    x = torch.from_numpy(((np.asarray(im, np.float32)/255 - IM_M)/IM_S).transpose(2,0,1))[None].to(dev)
    with torch.no_grad():
        c = ours.text_embed([intent], dev)
        s = ours.readout_forward(ours.visual_features(x), c)
        pr = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear", align_corners=False))[0,0].cpu().numpy()
    return gaussian_filter(pr, 1)

# --- gazeformer (sigma=70 steelman) ---
sys.path.insert(0, "gazeformer_repo")
from models import Transformer
from gazeformer import gazeformer as gf_model
from test import run_model
gdev = torch.device("cpu")
tr = Transformer(num_encoder_layers=6, nhead=8, d_model=512, num_decoder_layers=6,
                 dim_feedforward=512, img_hidden_dim=2048, lm_dmodel=768, device=gdev).to(gdev)
gf = gf_model(transformer=tr, spatial_dim=(20, 32), max_len=7, device=gdev).to(gdev)
gf.load_state_dict(torch.load("gazeformer_assets/checkpoints/gazeformer_cocosearch_TP.pkg", map_location=gdev)["model"])
gf.eval()
emb = np.load("gazeformer_assets/dataset/embeddings.npy", allow_pickle=True).item()
def gf_pred(task):
    feats = torch.load(f"gazeformer_assets/dataset/image_features/{task.replace(' ','_')}/{IMG_BASE.replace('jpg','pth')}", map_location="cpu").unsqueeze(0)
    sps = run_model(model=gf, src=feats, task=emb[task], device=gdev, num_samples=10)
    dens = np.zeros((EVAL, EVAL))
    for sp in sps:
        for y, x, t in sp:
            dens[min(int(y/320*EVAL), EVAL-1), min(int(x/512*EVAL), EVAL-1)] += 1
    return gaussian_filter(dens, (70*EVAL/DISP_H, 70*EVAL/DISP_W))

# --- HAT from corrected raw points (sigma=60 steelman) ---
raw = json.load(open("results/hat_raw_points.json"))
def hat_pred(task):
    dens = np.zeros((EVAL, EVAL))
    for row, col, cnt in raw[IMG_BASE + "|" + task]:
        fx = row / DISP_H; fy = col / DISP_W   # undo transpose
        dens[min(int(fy*EVAL), EVAL-1), min(int(fx*EVAL), EVAL-1)] += cnt
    return gaussian_filter(dens, (60*EVAL/DISP_H, 60*EVAL/DISP_W))

def load_map(path):
    m = np.asarray(Image.open(path).convert("L").resize((EVAL, EVAL), Image.BILINEAR), np.float64)
    return m

fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.0))
cols = ["Original", "Ground truth", "Gazeformer", "HAT", "xSoyelNet-Intent"]
for r, task in enumerate(TASKS):
    e = entries[task]
    orig = np.asarray(Image.open(e["image"]).convert("RGB").resize((EVAL, EVAL), Image.BILINEAR))
    row_imgs = [ (orig, None), (load_map(e["gt"]), "hot"), (gf_pred(task), "hot"),
                 (hat_pred(task), "hot"), (ours_pred(e["image"], e["intent"]), "hot") ]
    for c, (img, cmap) in enumerate(row_imgs):
        ax = axes[r, c]
        ax.imshow(img, cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(f'"find the {task}"', fontsize=11)
        if r == 0:
            ax.set_title(cols[c], fontsize=11)
fig.tight_layout()
fig.savefig("figures/fig_conditioning.png", dpi=200)
print("saved")
