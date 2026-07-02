"""
Retrain IntentNet with the v3 upgrades (448 input, ASPP-FiLM readout) on
COCO-Search18 target-present, and evaluate against the number that killed the
SPL submission: Gazeformer CC 0.738 on the same val split (v2 scored 0.707).

Protocol matches this week's kfold scripts: frozen backbone features cached once
(fp16), KL + (1-CC) loss on a 64x64 grid, AdamW, 60 epochs; evaluation at 224
(matching the v2/Gazeformer comparison resolution) with a Gaussian-finalizer
sigma selected on TRAINING data only (grid includes 0). Text embeddings for the
18 intents are precomputed once.

  PYTHONPATH=. .venv/bin/python train_intent_v3.py
"""
import json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

from model import IntentASPP

TRAIN_JSON = "data/cocosearch18/proc/intents_train.json"
VAL_JSON = "data/cocosearch18/proc/intents_val.json"
OUT_CKPT = "results/intent_v3.pt"
OUT_JSON = "results/intent_v3_results.json"

SIZE = 448
EVAL = 224          # match the v2 / Gazeformer comparison resolution
LOSSG = 64
EPOCHS = 60
BS = 16
LR = 3e-4
SIGMAS = [0, 1, 2, 3, 4, 6]
IM_M = np.array([0.485, 0.456, 0.406], np.float32)
IM_S = np.array([0.229, 0.224, 0.225], np.float32)


def cc_np(p, g):
    a, b = p - p.mean(), g - g.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def load_map(path, size):
    m = np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), np.float64)
    m = m - m.min(); s = m.sum()
    return m / s if s > 0 else m + 1.0 / m.size


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {dev}", flush=True)
    torch.manual_seed(0); np.random.seed(0)

    model = IntentASPP().to(dev)

    train = json.load(open(TRAIN_JSON))
    val = json.load(open(VAL_JSON))
    print(f"train pairs: {len(train)}  val pairs: {len(val)}", flush=True)

    # --- precompute the 18 intent text embeddings ---
    intents = sorted({e["intent"] for e in train} | {e["intent"] for e in val})
    with torch.no_grad():
        temb = model.text_embed(intents, dev)          # 18, 512
    t_idx = {s: i for i, s in enumerate(intents)}
    print(f"{len(intents)} unique intents embedded", flush=True)

    # --- cache backbone features per unique image (fp16) ---
    img_paths = sorted({e["image"] for e in train} | {e["image"] for e in val})
    feats = {}
    with torch.no_grad():
        for i, p in enumerate(img_paths):
            im = Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
            x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
            feats[p] = model.visual_features(x)[0].half().cpu()
            if (i + 1) % 300 == 0:
                print(f"  cached {i + 1}/{len(img_paths)}", flush=True)

    # --- per-pair targets ---
    gtl = [torch.tensor(load_map(e["gt"], LOSSG), dtype=torch.float32) for e in train]
    gte = [load_map(e["gt"], EVAL) for e in val]
    print("targets loaded", flush=True)

    params = [p for n, p in model.named_parameters()
              if not (n.startswith("vit.") or n.startswith("text_enc."))]
    n_tr = sum(p.numel() for p in params)
    print(f"trainable params: {n_tr/1e6:.2f}M", flush=True)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    @torch.no_grad()
    def predict_val():
        model.eval()
        preds = []
        for e in val:
            f = feats[e["image"]].float()[None].to(dev)
            c = temb[t_idx[e["intent"]]][None]
            s = model.readout_forward(f, c)
            pr = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear",
                                              align_corners=False))[0, 0].cpu().numpy().astype(np.float64)
            preds.append(pr)
        return preds

    @torch.no_grad()
    def select_sigma_on_train(n_sub=200):
        model.eval()
        rng = np.random.default_rng(0)
        sub = rng.choice(len(train), size=min(n_sub, len(train)), replace=False)
        preds, gts = [], []
        for j in sub:
            e = train[j]
            f = feats[e["image"]].float()[None].to(dev)
            c = temb[t_idx[e["intent"]]][None]
            s = model.readout_forward(f, c)
            pr = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear",
                                              align_corners=False))[0, 0].cpu().numpy().astype(np.float64)
            preds.append(pr); gts.append(load_map(e["gt"], EVAL))
        best_s, best_cc = 0, -2
        for s in SIGMAS:
            m = float(np.mean([cc_np(gaussian_filter(p, s) if s > 0 else p, g)
                                for p, g in zip(preds, gts)]))
            if m > best_cc:
                best_cc, best_s = m, s
        return best_s

    rng = np.random.default_rng(0)
    order = np.arange(len(train))
    best_val_cc, history = -1, []
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(order)
        for j0 in range(0, len(order), BS):
            ids = order[j0:j0 + BS]
            f = torch.stack([feats[train[j]["image"]].float() for j in ids]).to(dev)
            c = temb[[t_idx[train[j]["intent"]] for j in ids]]
            g = torch.stack([gtl[j] for j in ids]).to(dev)
            s = model.readout_forward(f, c)
            s = F.interpolate(s, size=(LOSSG, LOSSG), mode="bilinear", align_corners=False)[:, 0]
            logp = F.log_softmax(s.reshape(len(ids), -1), 1).reshape(len(ids), LOSSG, LOSSG)
            kl = (g * (torch.log(g + 1e-12) - logp)).sum((1, 2)).mean()
            ps = torch.sigmoid(s)
            pc = ps - ps.mean((1, 2), keepdim=True); gc = g - g.mean((1, 2), keepdim=True)
            ccb = (pc * gc).sum((1, 2)) / (torch.sqrt((pc ** 2).sum((1, 2)) * (gc ** 2).sum((1, 2))) + 1e-8)
            loss = kl + (1 - ccb.mean())
            opt.zero_grad(); loss.backward(); opt.step()

        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            sig = select_sigma_on_train()
            preds = predict_val()
            ccs = [cc_np(gaussian_filter(p, sig) if sig > 0 else p, g)
                   for p, g in zip(preds, gte)]
            vcc = float(np.mean(ccs))
            history.append({"epoch": ep + 1, "sigma": sig, "val_cc": vcc})
            print(f"epoch {ep+1}: sigma={sig}  val CC={vcc:.4f}   "
                  f"(v2 was 0.707, Gazeformer 0.738)", flush=True)
            if vcc > best_val_cc:
                best_val_cc = vcc
                torch.save({k: v for k, v in model.state_dict().items()
                            if not (k.startswith("vit.") or k.startswith("text_enc."))}, OUT_CKPT)

    json.dump({"history": history, "best_val_cc": best_val_cc,
               "reference": {"v2_cc": 0.707, "gazeformer_cc": 0.738}},
              open(OUT_JSON, "w"), indent=2)
    print(f"\nBEST val CC = {best_val_cc:.4f}  (saved readout -> {OUT_CKPT})", flush=True)


if __name__ == "__main__":
    main()
