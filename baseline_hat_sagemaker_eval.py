"""
SageMaker entrypoint: evaluate HAT (CVPR'24) as a density-map predictor on our
COCO-Search18 val protocol, mirroring the Gazeformer eval exactly:
  - 10 sampled scanpaths per (image, task) pair (sample_action=True),
  - fixations mapped to 1680x1050 display space, sigma=40 Gaussian (identical to
    GT construction), resized to 224, scored with CC (+ NSS/AUC on human
    fixation points), full val and clean val.

Runtime setup (inside the SM PyTorch 1.13.1/cu117 container):
  - pip install detectron2 prebuilt wheel (torch1.13/cu117) + wget
  - clone HAT, compile MSDeformAttn ops, download HAT_TP.pt + M2F weights.
"""
import json, os, subprocess, sys

def sh(cmd, **kw):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)

DATA = os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data")
OUT = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

# ---------- environment ----------
sh("pip install --no-cache-dir wget 'numpy<2' scipy timm==0.6.12 omegaconf hydra-core 'pillow<10'")
sh("python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@v0.6'")
sh("git clone --depth 1 https://github.com/cvlab-stonybrook/HAT.git /opt/ml/code/HAT")
# The released M2F_R50.pkl uses mid-2022 d2 dev naming (stages.res*); d2 v0.6's ResNet
# expects plain res* keys. Reverse HAT's (no-op) remap into a stages.-prefix STRIP.
sh("""sed -i "s/if k\\[:3\\] == 'res':/if k[:7] == 'stages.':/" /opt/ml/code/HAT/hat/models.py""")
sh("""sed -i 's/bb_weights_new\\["stages." + k\\] = v/bb_weights_new[k[7:]] = v/' /opt/ml/code/HAT/hat/models.py""")
sh("grep -n 'stages.' /opt/ml/code/HAT/hat/models.py | head -5")
os.chdir("/opt/ml/code/HAT")
sh("cd hat/pixel_decoder/ops && TORCH_CUDA_ARCH_LIST='7.5' pip install --no-build-isolation .", env={**os.environ})
sh("python -c 'import torch, MultiScaleDeformableAttention; print(\"MSDA import OK\")'")
os.makedirs("checkpoints", exist_ok=True); os.makedirs("pretrained_models", exist_ok=True)
sh("cd checkpoints && wget -q http://vision.cs.stonybrook.edu/~cvlab_download/HAT/HAT_TP.pt")
sh("cd pretrained_models && wget -q http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50_MSDeformAttnPixelDecoder.pkl "
   "&& wget -q http://vision.cs.stonybrook.edu/~cvlab_download/HAT/pretrained_models/M2F_R50.pkl")

sys.path.insert(0, "/opt/ml/code/HAT")
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from hat.models import HumanAttnTransformer
from hat.evaluation import scanpath_decode
from common.config import JsonConfig

DISP_W, DISP_H, EVAL, SIGMA_GT, NUM_SAMPLES = 1680, 1050, 224, 40, 10
dev = "cuda" if torch.cuda.is_available() else "cpu"

hparams = JsonConfig("configs/coco_search18_dense_SSL_TP.json")
model = HumanAttnTransformer(
    hparams.Data, num_decoder_layers=hparams.Model.n_dec_layers,
    hidden_dim=hparams.Model.embedding_dim, nhead=hparams.Model.n_heads,
    ntask=18, tgt_vocab_size=hparams.Data.patch_count + len(hparams.Data.special_symbols),
    num_output_layers=hparams.Model.num_output_layers,
    separate_fix_arch=hparams.Model.separate_fix_arch,
    train_encoder=hparams.Train.train_backbone,
    train_pixel_decoder=hparams.Train.train_pixel_decoder,
    use_dino=hparams.Train.use_dino_pretrained_model,
    dropout=hparams.Train.dropout, dim_feedforward=hparams.Model.hidden_dim,
    parallel_arch=hparams.Model.parallel_arch, dorsal_source=hparams.Model.dorsal_source,
    num_encoder_layers=hparams.Model.n_enc_layers,
    output_centermap="centermap_pred" in hparams.Train.losses,
    output_saliency="saliency_pred" in hparams.Train.losses,
    output_target_map="target_map_pred" in hparams.Train.losses,
    transfer_learning_setting=hparams.Train.transfer_learn,
    project_queries=hparams.Train.project_queries, is_pretraining=False,
    output_feature_map_name=hparams.Model.output_feature_map_name).to(dev)
ckpt = torch.load("checkpoints/HAT_TP.pt", map_location=dev)
# HAT_TP.pt was trained under mid-2022 d2 naming (backbone.stages.res*); our v0.6-built
# model uses backbone.res* -- strip the stages. segment.
sd_ckpt = {k.replace("backbone.stages.", "backbone."): v for k, v in ckpt["model"].items()}
model.load_state_dict(sd_ckpt)
model.eval()
preset_tasks = np.load("demo/all_task_ids.npy", allow_pickle=True).item()
print("HAT loaded", flush=True)

size = (hparams.Data.im_h, hparams.Data.im_w)
transform = transforms.Compose([
    transforms.Resize(size=size), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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
    order = np.argsort(fp)
    return float(np.trapz(tp[order], fp[order]))

manifest = json.load(open(f"{DATA}/manifest.json"))
fixjson = json.load(open(f"{DATA}/fixations.json"))
pairs, train_images = manifest["pairs"], set(manifest["train_images"])
torch.manual_seed(42); np.random.seed(42)

ccs, nsss, aucs, is_clean = [], [], [], []
raw_points = {}
for n, e in enumerate(pairs):
    img = Image.open(f"{DATA}/images/{e['image']}").convert("RGB")
    x = transform(img).unsqueeze(0).to(dev)
    task_id = torch.tensor([preset_tasks[e["task"]]], dtype=torch.long).to(dev)
    dens = np.zeros((DISP_H, DISP_W), np.float32)
    for _ in range(NUM_SAMPLES):
        normalized_sp, _ = scanpath_decode(model, x, task_id, hparams.Data,
                                            sample_action=True, center_initial=True)
        fixs = normalized_sp[0].cpu().numpy()   # normalized (y, x) in [0,1]
        for fy, fx in fixs:
            xi = int(np.clip(fx * DISP_W, 0, DISP_W - 1))
            yi = int(np.clip(fy * DISP_H, 0, DISP_H - 1))
            dens[yi, xi] += 1
    raw_points[e["image"] + "|" + e["task"]] = np.argwhere(dens > 0).tolist() if False else None
    dens_pts = np.argwhere(dens > 0)
    raw_points[e["image"] + "|" + e["task"]] = [[int(r), int(c), int(dens[r, c])] for r, c in dens_pts]
    dens = gaussian_filter(dens, SIGMA_GT)
    pr = np.asarray(Image.fromarray(dens).resize((EVAL, EVAL), Image.BILINEAR), np.float64)
    gt = load_map(f"{DATA}/maps/{e['gt']}", EVAL)
    ccs.append(cc_np(pr, gt))
    key = e["image"] + "|" + e["task"]
    if key in fixjson:
        fmap = np.zeros((EVAL, EVAL))
        for fx, fy in zip(fixjson[key]["X"], fixjson[key]["Y"]):
            fmap[int(np.clip(fy*EVAL/DISP_H, 0, EVAL-1)), int(np.clip(fx*EVAL/DISP_W, 0, EVAL-1))] = 1
        nsss.append(NSS(pr, fmap)); aucs.append(AUC_Judd(pr, fmap))
    else:
        nsss.append(np.nan); aucs.append(np.nan)
    is_clean.append(e["image"] not in train_images)
    if (n+1) % 50 == 0:
        print(f"  {n+1}/{len(pairs)} running CC={np.mean(ccs):.4f}", flush=True)

ccs = np.array(ccs); clean = np.array(is_clean)
res = {"full_val_cc": float(np.nanmean(ccs)), "clean_val_cc": float(np.nanmean(ccs[clean])),
       "full_val_nss": float(np.nanmean(nsss)), "full_val_auc": float(np.nanmean(aucs)),
       "num_samples": NUM_SAMPLES, "sigma_gt": SIGMA_GT, "n_pairs": len(pairs)}
print(json.dumps(res, indent=2), flush=True)
json.dump(res, open(f"{OUT}/hat_density_our_val.json", "w"), indent=2)
json.dump(raw_points, open(f"{OUT}/hat_raw_points.json", "w"))
print("done", flush=True)
