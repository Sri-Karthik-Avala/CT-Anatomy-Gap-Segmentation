# made by - Karthik

import os
import sys
import math
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


SEED        = _env("SEED", 1337, int)
IMG_SIZE    = 128
N_CLASSES   = 9
N_FG        = 8
MODEL       = _env("MODEL", "deeplab", str)
N_FOLDS     = _env("N_FOLDS", 5, int)
EPOCHS      = _env("EPOCHS", 55, int)
BATCH       = _env("BATCH", 32, int)
LR          = _env("LR", 1e-3, float)
WD          = _env("WD", 1e-4, float)
BASE_CH     = _env("BASE_CH", 32, int)
N_REPEAT    = _env("N_REPEAT", 1, int)
WARMUP      = _env("WARMUP", 5, int)
USE_TTA     = _env("USE_TTA", 1, int)
TIME_BUDGET = _env("TIME_BUDGET", 1000, int)

CLASS_NAMES = ["lung_airway", "cardiomediastinal", "hepatosplenic", "renal_adrenal",
               "bowel_pancreas", "pelvic_urinary", "axial_bone", "appendicular_muscle"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (DEVICE.type == "cuda")
if not USE_CUDA:
    if os.environ.get("EPOCHS") is None:
        EPOCHS = 25
    if os.environ.get("N_FOLDS") is None:
        N_FOLDS = 3
    if os.environ.get("N_REPEAT") is None:
        N_REPEAT = 1

MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.25, 0.25, 0.25], dtype=np.float32)


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)


def find_data_root():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = ["dataset/public", ".", "input", "dataset", "/kaggle/input",
             os.path.join(here, "dataset", "public"), here,
             os.path.join(here, "..", "dataset", "public"),
             os.path.join(here, "..")]
    for c in cands:
        if os.path.exists(os.path.join(c, "train.csv")) and \
           os.path.exists(os.path.join(c, "test.csv")):
            return c
    for base in [".", here, os.path.join(here, "..")]:
        for root, _, files in os.walk(base):
            if "train.csv" in files and "test.csv" in files:
                return root
    raise FileNotFoundError("Could not locate train.csv / test.csv")


DATA_ROOT = find_data_root()
OUT_DIR = "working"
os.makedirs(OUT_DIR, exist_ok=True)


def rel(p):
    return os.path.join(DATA_ROOT, p)


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path):
    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m.astype(np.uint8)


def rle_encode(mask):
    pixels = mask.flatten(order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


class ArrDS(Dataset):
    def __init__(self, imgs, masks, idx):
        self.imgs = imgs
        self.masks = masks
        self.idx = np.asarray(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        img = self.imgs[j].astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        if self.masks is not None:
            mask = torch.from_numpy(self.masks[j].astype(np.int64))
        else:
            mask = torch.zeros(IMG_SIZE, IMG_SIZE, dtype=torch.long)
        return img, mask


def gpu_augment(img, mask):
    B = img.shape[0]
    dev = img.device
    flip = torch.rand(B, device=dev) < 0.5
    if flip.any():
        img[flip] = torch.flip(img[flip], dims=[3])
        mask[flip] = torch.flip(mask[flip], dims=[2])
    ang = (torch.rand(B, device=dev) * 2 - 1) * (12.0 * math.pi / 180.0)
    scale = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * 0.12
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.06
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.06
    cos = torch.cos(ang) / scale
    sin = torch.sin(ang) / scale
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    m = F.grid_sample(mask.unsqueeze(1).float(), grid, mode="nearest",
                      padding_mode="zeros", align_corners=False)
    mask = m.squeeze(1).long()
    if torch.rand(1).item() < 0.5:
        b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.12
        c = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.12
        img = img * c + b
    return img, mask


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=N_CLASSES, base=BASE_CH):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc1 = DoubleConv(in_ch, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.bott = DoubleConv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4 = DoubleConv(c[4], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], n_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bott(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                      nn.BatchNorm2d(cout))

    def forward(self, x):
        idn = x if self.down is None else self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + idn, inplace=True)


def make_layer(cin, cout, n, stride=1, dilation=1):
    layers = [BasicBlock(cin, cout, stride=stride, dilation=dilation)]
    for _ in range(n - 1):
        layers.append(BasicBlock(cout, cout, dilation=dilation))
    return nn.Sequential(*layers)


class ASPP(nn.Module):
    def __init__(self, cin, cout=256, rates=(1, 2, 4, 6)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(cin, cout, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True)))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(cin, cout, 1, bias=False),
                                  nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(cout * (len(rates) + 1), cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Dropout(0.3))

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        p = self.pool(x)
        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(p)
        return self.project(torch.cat(feats, 1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_ch=3, n_classes=N_CLASSES, base=BASE_CH):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.layer1 = make_layer(base, base * 2, 2, stride=1)
        self.layer2 = make_layer(base * 2, base * 4, 2, stride=2)
        self.layer3 = make_layer(base * 4, base * 8, 2, stride=2)
        self.layer4 = make_layer(base * 8, base * 16, 2, stride=1, dilation=2)
        self.aspp = ASPP(base * 16, 256, rates=(1, 2, 4, 6))
        self.low_proj = nn.Sequential(nn.Conv2d(base * 2, 48, 1, bias=False),
                                      nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(256, n_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        s = self.stem(x)
        l1 = self.layer1(s)
        x = self.layer2(l1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=l1.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(l1)
        x = torch.cat([x, low], 1)
        x = self.decoder(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(x)


def build_model():
    if MODEL == "deeplab":
        return DeepLabV3Plus().to(DEVICE)
    return UNet().to(DEVICE)


def compute_class_weights(masks):
    counts = np.ones(N_CLASSES, dtype=np.float64)
    counts += np.bincount(masks.flatten(), minlength=N_CLASSES)
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(freq)
    w = w / w.mean()
    w = np.clip(w, 0.3, 5.0)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


class TverskyLossMC(nn.Module):
    def __init__(self, alpha=0.4, beta=0.6, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.softmax(logits, dim=1)[:, 1:]
        oh = F.one_hot(target, N_CLASSES).permute(0, 3, 1, 2).float()[:, 1:]
        dims = (0, 2, 3)
        tp = (prob * oh).sum(dims)
        fp = (prob * (1 - oh)).sum(dims)
        fn = ((1 - prob) * oh).sum(dims)
        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tv.mean()


class ComboLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.tversky = TverskyLossMC(alpha=0.4, beta=0.6)

    def forward(self, logits, target):
        return self.ce(logits, target) + self.tversky(logits, target)


def erode4(p):
    e = p.copy()
    up = np.zeros_like(p); up[:, 1:, :] = p[:, :-1, :]
    dn = np.zeros_like(p); dn[:, :-1, :] = p[:, 1:, :]
    lf = np.zeros_like(p); lf[:, :, 1:] = p[:, :, :-1]
    rt = np.zeros_like(p); rt[:, :, :-1] = p[:, :, 1:]
    return e & up & dn & lf & rt


def boundary(p):
    return p & (~erode4(p))


def precompute_true_parts(true_labels):
    parts = []
    for c in range(1, N_CLASSES):
        T = (true_labels == c)
        ts = T.reshape(T.shape[0], -1).sum(1)
        tb = boundary(T)
        parts.append((T, ts, tb, bool(ts.sum() > 0)))
    return parts


def composite_from_parts(pred, true_parts):
    eps = 1e-7
    N = pred.shape[0]
    HW = IMG_SIZE * IMG_SIZE
    class_inter = np.zeros(N_FG); class_denom = np.zeros(N_FG)
    class_has_true = np.zeros(N_FG, dtype=bool)
    cc_sum = 0.0; cc_n = 0
    bTP = bFP = bFN = 0
    area_sum = 0.0; area_n = 0
    for c in range(1, N_CLASSES):
        T, ts, tb, has_t = true_parts[c - 1]
        P = (pred == c)
        ps = P.reshape(N, -1).sum(1)
        inter = (P & T).reshape(N, -1).sum(1)
        class_inter[c - 1] = 2 * inter.sum()
        class_denom[c - 1] = ps.sum() + ts.sum()
        class_has_true[c - 1] = has_t
        m = ts > 0
        if m.any():
            cc = (2 * inter[m]) / (ps[m] + ts[m] + eps)
            cc_sum += cc.sum(); cc_n += int(m.sum())
        pb = boundary(P)
        bTP += int((pb & tb).sum())
        bFP += int((pb & (~tb)).sum())
        bFN += int(((~pb) & tb).sum())
        area_sum += float((np.abs(ps - ts) / HW).sum()); area_n += N
    per_class = class_inter[class_has_true] / (class_denom[class_has_true] + eps)
    mean_class_dice = per_class.mean() if per_class.size else 0.0
    mean_case_class_dice = (cc_sum / cc_n) if cc_n else 0.0
    boundary_f1 = (2 * bTP) / (2 * bTP + bFP + bFN + eps)
    area_score = max(0.0, 1.0 - 12.0 * (area_sum / max(area_n, 1)))
    quality = (0.45 * mean_class_dice + 0.30 * mean_case_class_dice +
               0.20 * boundary_f1 + 0.05 * area_score)
    final = float(np.clip(quality ** 2, 0.0, 1.0))
    return dict(final=final, quality=quality, mean_class_dice=mean_class_dice,
                mean_case_class_dice=mean_case_class_dice, boundary_f1=boundary_f1,
                area_score=area_score)


@torch.no_grad()
def predict_probs(model, loader, tta=False):
    model.eval()
    out = []
    for img, _ in loader:
        img = img.to(DEVICE, non_blocking=True)
        prob = torch.softmax(model(img).float(), dim=1)
        if tta:
            pf = torch.softmax(model(torch.flip(img, dims=[3])).float(), dim=1)
            prob = (prob + torch.flip(pf, dims=[3])) / 2.0
        out.append(prob.cpu())
    return torch.cat(out, 0).numpy()


def group_kfold(volume_ids, n_splits, seed):
    uniq = sorted(set(volume_ids))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(uniq))
    uniq = [uniq[i] for i in order]
    bins = [[] for _ in range(n_splits)]
    for i, v in enumerate(uniq):
        bins[i % n_splits].append(v)
    vols = np.asarray(volume_ids)
    splits = []
    for f in range(n_splits):
        val_vols = set(bins[f])
        va = np.where(np.isin(vols, list(val_vols)))[0]
        tr = np.where(~np.isin(vols, list(val_vols)))[0]
        splits.append((tr, va))
    return splits


def make_scheduler(opt):
    def fn(ep):
        if ep < WARMUP:
            return (ep + 1) / max(WARMUP, 1)
        t = (ep - WARMUP) / max(EPOCHS - WARMUP, 1)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train_one(rep, fold, train_imgs, train_masks, tr_idx, va_idx, class_weights, deadline):
    seed_everything(SEED + fold + 100 * rep)
    t_fold = time.time()
    print(f"  [rep{rep} fold{fold}] training on {len(tr_idx)} imgs...", flush=True)
    model = build_model()
    tr_ds = ArrDS(train_imgs, train_masks, tr_idx)
    va_ds = ArrDS(train_imgs, train_masks, va_idx)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                       drop_last=True, pin_memory=USE_CUDA)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0,
                       pin_memory=USE_CUDA)
    va_parts = precompute_true_parts(train_masks[va_idx])

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = make_scheduler(opt)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_CUDA)
    criterion = ComboLoss(class_weights)

    best_score, best_state = -1.0, None
    val_start = int(EPOCHS * 0.4)
    for ep in range(EPOCHS):
        model.train()
        for img, mask in tr_ld:
            img = img.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            img, mask = gpu_augment(img, mask)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_CUDA):
                logit = model(img)
                loss = criterion(logit, mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        do_val = (ep % 4 == 0) or ep >= EPOCHS - 8 or ep >= val_start
        if do_val:
            probs = predict_probs(model, va_ld, tta=False)
            s = composite_from_parts(probs.argmax(1).astype(np.uint8), va_parts)["final"]
            if s > best_score:
                best_score = s
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if time.time() > deadline:
            print(f"  [rep{rep} fold{fold}] stopping at epoch {ep} (time budget)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    va_probs = predict_probs(model, va_ld, tta=bool(USE_TTA))
    print(f"  [rep{rep} fold{fold}] best val final = {best_score:.4f}  ({time.time()-t_fold:.0f}s)",
          flush=True)
    return model, va_probs


def tune_bias(oof_probs, true_parts):
    logp = np.log(np.clip(oof_probs, 1e-7, 1.0)).astype(np.float32)
    logp = np.ascontiguousarray(np.transpose(logp, (0, 2, 3, 1)))
    buf = np.empty_like(logp)
    b9 = np.zeros(N_CLASSES, dtype=np.float32)
    bias = np.zeros(N_FG, dtype=np.float32)

    def score(b):
        b9[1:] = b
        np.add(logp, b9, out=buf)
        return composite_from_parts(buf.argmax(-1).astype(np.uint8), true_parts)["final"]

    base = score(bias)
    grid = [-3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5]
    for _ in range(2):
        for c in range(N_FG):
            best_b, best_s = bias[c], score(bias)
            for v in grid:
                trial = bias.copy(); trial[c] = v
                s = score(trial)
                if s > best_s + 1e-5:
                    best_s, best_b = s, v
            bias[c] = best_b
    tuned = score(bias)
    print(f"  [tune] bias = {np.round(bias,2).tolist()}", flush=True)
    print(f"  [tune] OOF final {base:.4f} -> {tuned:.4f}", flush=True)
    return bias


def labels_from_bias(probs, bias):
    logp = np.log(np.clip(probs, 1e-7, 1.0))
    logp[:, 1:] += bias[None, :, None, None]
    return logp.argmax(1).astype(np.uint8)


def write_submission(test_labels, test_df):
    rows = []
    for i, case_id in enumerate(test_df["case_id"].values):
        lab = test_labels[i]
        row = {"case_id": case_id}
        for c, name in enumerate(CLASS_NAMES, start=1):
            row[name] = rle_encode((lab == c).astype(np.uint8))
        rows.append(row)
    sub = pd.DataFrame(rows, columns=["case_id"] + CLASS_NAMES)
    sub.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
    sub.to_csv("submission.csv", index=False)
    return sub


def main():
    global MEAN, STD
    t0 = time.time()
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = True
    print(f"[setup] device={DEVICE} model={MODEL} root={DATA_ROOT} folds={N_FOLDS} "
          f"epochs={EPOCHS} base={BASE_CH} repeat={N_REPEAT}", flush=True)

    train_df = pd.read_csv(rel("train.csv"))
    test_df = pd.read_csv(rel("test.csv"))
    print(f"[data] train={len(train_df)} test={len(test_df)} "
          f"volumes={train_df['volume_id'].nunique()}", flush=True)

    train_imgs = np.stack([load_image(rel(p)) for p in train_df["image_path"].values])
    train_masks = np.stack([load_mask(rel(p)) for p in train_df["mask_path"].values])
    test_imgs = np.stack([load_image(rel(p)) for p in test_df["image_path"].values])

    flat = train_imgs.reshape(-1, 3).astype(np.float32) / 255.0
    MEAN = flat.mean(0)
    STD = flat.std(0) + 1e-6
    print(f"[data] mean={np.round(MEAN,3).tolist()} std={np.round(STD,3).tolist()}", flush=True)

    class_weights = compute_class_weights(train_masks)
    print(f"[data] class weights = {np.round(class_weights.cpu().numpy(),2).tolist()}", flush=True)

    n_tr = len(train_df)
    oof_probs = np.zeros((n_tr, N_CLASSES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    oof_filled = np.zeros(n_tr, dtype=bool)
    test_probs = np.zeros((len(test_df), N_CLASSES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    splits = group_kfold(train_df["volume_id"].values, N_FOLDS, SEED)
    test_ds = ArrDS(test_imgs, None, np.arange(len(test_df)))
    test_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0,
                         pin_memory=USE_CUDA)

    deadline = t0 + TIME_BUDGET - 200
    zero_bias = np.zeros(N_FG, dtype=np.float32)
    n_models = 0
    reps_done = 0
    stop = False
    for rep in range(N_REPEAT):
        for fold, (tr_idx, va_idx) in enumerate(splits):
            model, va_probs = train_one(rep, fold, train_imgs, train_masks,
                                        tr_idx, va_idx, class_weights, deadline)
            oof_probs[va_idx] += va_probs
            oof_filled[va_idx] = True
            test_probs += predict_probs(model, test_ld, tta=bool(USE_TTA))
            n_models += 1
            del model
            if USE_CUDA:
                torch.cuda.empty_cache()
            prov = labels_from_bias(test_probs / max(n_models, 1), zero_bias)
            write_submission(prov, test_df)
            print(f"  [save] provisional submission after {n_models} models", flush=True)
            if time.time() > deadline:
                print(f"[budget] hit time budget after {n_models} models", flush=True)
                stop = True
                break
        if stop:
            break
        reps_done += 1

    test_probs /= max(n_models, 1)

    bias = np.zeros(N_FG, dtype=np.float32)
    if oof_filled.all() and reps_done >= 1:
        oof_probs /= max(reps_done, 1)
        true_parts = precompute_true_parts(train_masks)
        bias = tune_bias(oof_probs, true_parts)
        oof_labels = labels_from_bias(oof_probs, bias)
        m = composite_from_parts(oof_labels, true_parts)
        print("[OOF] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)

    test_labels = labels_from_bias(test_probs, bias)
    sub = write_submission(test_labels, test_df)
    print(f"[done] wrote submission with {len(sub)} rows in {time.time()-t0:.0f}s", flush=True)
    assert len(sub) == len(test_df), "row count mismatch"
    print("[check] submission OK", flush=True)


if __name__ == "__main__":
    main()
