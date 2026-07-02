"""
Phase-2 data: COCO-Search18 target-present scanpaths -> per-(image, target) fixation
density maps + intent "find the {target}".

Each (image, task) aggregates fixations from all subjects searching for that target.
Display space is 1680x1050; images in the TP zip are at that resolution.

Output:
  data/cocosearch18/proc/images/<imgstem>.jpg
  data/cocosearch18/proc/maps/<imgstem>__<task>.png
  data/cocosearch18/proc/intents_{train,val}.json   [{image, gt, fix, intent, task}, ...]
"""
import os, json, glob
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = "data/cocosearch18"   # EDIT ME: dir with the COCO-Search18 TP images + fixation jsons
IMG_DIRS = [f"{ROOT}/images", f"{ROOT}/coco_search18_images_TP", f"{ROOT}/images-TP"]
OUT = f"{ROOT}/proc"
DISP_W, DISP_H = 1680, 1050


def find_image(name):
    for d in IMG_DIRS:
        for sub in ("", *[p for p in glob.glob(d + "/*") if os.path.isdir(p)]):
            p = os.path.join(sub or d, name)
            if os.path.exists(p):
                return p
    hits = glob.glob(f"{ROOT}/**/{name}", recursive=True)
    return hits[0] if hits else None


def density(xs, ys, W, H, sigma=40):
    m = np.zeros((H, W), np.float32)
    xs = np.clip(np.asarray(xs), 0, W - 1).astype(int)
    ys = np.clip(np.asarray(ys), 0, H - 1).astype(int)
    np.add.at(m, (ys, xs), 1)
    m = gaussian_filter(m, sigma=sigma)
    mx = m.max()
    return (m / mx * 255).astype(np.uint8) if mx > 0 else m.astype(np.uint8)


def build(split_files, out_json, min_fix=10):
    trials = []
    for sf in split_files:
        trials += json.load(open(sf))
    # group fixations by (image, task)
    grp = defaultdict(lambda: {"x": [], "y": [], "task": None})
    for t in trials:
        if t.get("condition") != "present":
            continue
        key = (t["name"], t["task"])
        grp[key]["x"] += list(t["X"]); grp[key]["y"] += list(t["Y"]); grp[key]["task"] = t["task"]

    os.makedirs(f"{OUT}/images", exist_ok=True)
    os.makedirs(f"{OUT}/maps", exist_ok=True)
    pairs = []; saved_img = set(); missing = 0
    for (name, task), d in grp.items():
        if len(d["x"]) < min_fix:
            continue
        ip = f"{ROOT}/images/{task}/{name}"
        if not os.path.exists(ip):
            ip = find_image(name)
        if ip is None or not os.path.exists(ip):
            missing += 1; continue
        img = Image.open(ip).convert("RGB"); W, H = img.size
        # scale fixations if stored image differs from display space
        sx, sy = W / DISP_W, H / DISP_H
        xs = [x * sx for x in d["x"]]; ys = [y * sy for y in d["y"]]
        dens = density(xs, ys, W, H)
        stem = os.path.splitext(name)[0]
        if stem not in saved_img:
            img.save(f"{OUT}/images/{stem}.jpg", quality=92); saved_img.add(stem)
        mapname = f"{stem}__{task.replace(' ', '_')}.png"
        Image.fromarray(dens).save(f"{OUT}/maps/{mapname}")
        pairs.append({"image": f"{OUT}/images/{stem}.jpg",
                      "gt": f"{OUT}/maps/{mapname}", "fix": None,
                      "intent": f"find the {task}", "task": task})
    json.dump(pairs, open(out_json, "w"))
    print(f"{out_json}: {len(pairs)} (image,task) pairs, {len(saved_img)} images, {missing} missing-img skipped")
    return pairs


def main():
    train = build([f"{ROOT}/coco_search18_fixations_TP_train_split1.json",
                   f"{ROOT}/coco_search18_fixations_TP_train_split2.json"],
                  f"{OUT}/intents_train.json")
    val = build([f"{ROOT}/coco_search18_fixations_TP_validation_split1.json"],
                f"{OUT}/intents_val.json")
    from collections import Counter
    print("train tasks:", dict(Counter(p["task"] for p in train)))


if __name__ == "__main__":
    main()
