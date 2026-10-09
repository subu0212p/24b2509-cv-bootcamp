"""Shared utilities: data loading (kept on the GPU), GPU augmentation, optimizer/schedule, evaluation.

Training setup used by EVERY run (baseline and all methods, real and synthetic noise):
  * ResNet-18 from model.py, trained from scratch
  * SGD, momentum 0.9, Nesterov, weight decay 5e-4, initial LR 0.1
  * cosine LR schedule from 0.1 to 0 over the run (updated every iteration)
  * batch size 128
  * augmentation: random crop 32x32 with 4-pixel zero padding + random horizontal flip
  * mixed precision (fp16 autocast), seed 42
"""
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import ResNet18

NUM_CLASSES = 10
SEED = 42
BATCH_SIZE = 128
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 60


# ----------------------------------------------------------------------------- misc
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- data
class Data:
    """Loads the provided files. Images are kept as uint8 tensors on the device."""

    def __init__(self, data_dir, device, smoke=False):
        self.device = device
        self.data_dir = data_dir
        tr = np.load(os.path.join(data_dir, "train_images.npy"))
        te = np.load(os.path.join(data_dir, "test_images.npy"))
        assert tr.shape == (50000, 32, 32, 3) and te.shape == (10000, 32, 32, 3)
        self.train_idx = np.loadtxt(os.path.join(data_dir, "train_idx.txt"), dtype=np.int64)
        self.val_idx = np.loadtxt(os.path.join(data_dir, "val_idx.txt"), dtype=np.int64)
        noisy, synth, clean = np.zeros(50000, np.int64), np.zeros(50000, np.int64), np.zeros(50000, np.int64)
        with open(os.path.join(data_dir, "labels.csv")) as f:
            for row in csv.DictReader(f):
                i = int(row["index"])
                noisy[i], synth[i], clean[i] = int(row["noisy_label"]), int(row["synthetic_label"]), int(row["clean_label"])
        self.labels = {"noisy": noisy, "synthetic": synth, "clean": clean}
        if smoke:  # tiny subset for CPU testing of the pipeline only
            rs = np.random.RandomState(0)
            self.train_idx = np.sort(rs.choice(self.train_idx, 1200, replace=False))
            self.val_idx = np.sort(rs.choice(self.val_idx, 300, replace=False))
            te = te[:200]   # smoke only: tiny test set (predictions are tiled to 10k when writing)
        # normalisation statistics from the training split only
        sub = tr[self.train_idx].astype(np.float64) / 255.0
        self.mean = torch.tensor(sub.mean(axis=(0, 1, 2)), dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(sub.std(axis=(0, 1, 2)), dtype=torch.float32, device=device).view(1, 3, 1, 1)
        del sub
        self.X = torch.from_numpy(tr).permute(0, 3, 1, 2).contiguous().to(device)      # (50000,3,32,32) uint8
        self.Xte = torch.from_numpy(te).permute(0, 3, 1, 2).contiguous().to(device)    # (10000,3,32,32) uint8

    def norm(self, xb_uint8):
        x = xb_uint8.float().div_(255.0)
        return ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)

    def augment(self, xb_uint8):
        """Random crop (4px zero padding) + horizontal flip, done on the GPU, then normalise."""
        B = xb_uint8.shape[0]
        dev = xb_uint8.device
        x = F.pad(xb_uint8.float(), (4, 4, 4, 4))                      # zero (black) padding, like torchvision
        i = torch.randint(0, 9, (B,), device=dev)
        j = torch.randint(0, 9, (B,), device=dev)
        ar = torch.arange(32, device=dev)
        rows = (i[:, None] + ar)[:, None, :, None]                      # B,1,32,1
        cols = (j[:, None] + ar)[:, None, None, :]                      # B,1,1,32
        bidx = torch.arange(B, device=dev)[:, None, None, None]
        cidx = torch.arange(3, device=dev)[None, :, None, None]
        x = x[bidx, cidx, rows, cols]
        flip = torch.rand(B, device=dev) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        x = x.div_(255.0)
        return ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)


# ----------------------------------------------------------------------------- model / optim
def make_model(device):
    net = ResNet18(num_classes=NUM_CLASSES).to(device)
    return net.to(memory_format=torch.channels_last)


def make_optimizer(net):
    return torch.optim.SGD(net.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                           weight_decay=WEIGHT_DECAY, nesterov=True)


def cosine_lr(opt, progress_epochs, total_epochs):
    """progress_epochs is fractional (epoch + iter/num_iters)."""
    lr = 0.5 * BASE_LR * (1.0 + math.cos(math.pi * min(progress_epochs / total_epochs, 1.0)))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda"))


def make_scaler(device):
    return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


@torch.no_grad()
def predict_probs(nets, data, X, idx=None, bs=1000, tta=True):
    """Softmax probabilities (float32, CPU numpy). nets: a model or list of models (averaged).
    tta=True also averages the horizontally flipped image."""
    if not isinstance(nets, (list, tuple)):
        nets = [nets]
    for n in nets:
        n.eval()
    dev = data.device
    N = X.shape[0] if idx is None else len(idx)
    out = np.zeros((N, NUM_CLASSES), np.float32)
    idx_t = None if idx is None else torch.as_tensor(idx, device=dev)
    for s in range(0, N, bs):
        xb = X[s:s + bs] if idx_t is None else X[idx_t[s:s + bs]]
        x = data.norm(xb)
        p = 0
        with autocast(dev):
            for n in nets:
                p = p + F.softmax(n(x).float(), 1)
                if tta:
                    p = p + F.softmax(n(x.flip(3)).float(), 1)
        p = p / (len(nets) * (2 if tta else 1))
        out[s:s + bs] = p.cpu().numpy()
    return out


def acc(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean() * 100.0)


def write_pred_csv(path, preds):
    preds = np.asarray(preds).astype(int)
    if os.environ.get("SMOKE_TILE") == "1":
        preds = np.resize(preds, 10000)
    assert preds.shape == (10000,) and preds.min() >= 0 and preds.max() <= 9
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pred"])
        for i, p in enumerate(preds):
            w.writerow([i, int(p)])
