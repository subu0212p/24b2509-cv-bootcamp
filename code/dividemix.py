"""DivideMix (Li, Socher & Hoi, ICLR 2020) adapted to the bootcamp budget.

Two ResNet-18s (= 2 networks, the per-method maximum), each trained for 60 epochs total:
  * epochs 1..WARMUP: plain cross-entropy (+ confidence penalty) on all training labels
  * afterwards, every epoch and for each network:
      - the OTHER network computes the per-sample training loss; a 2-component Gaussian mixture on
        those losses gives p(clean) for every sample  ("co-divide")
      - samples with p(clean) > 0.5 form the labelled set, the rest are used WITHOUT labels
      - label co-refinement (labelled) / co-guessing (unlabelled), sharpening, MixMatch-style mixup
      - loss = Lx (soft CE) + lambda_u * Lu (MSE, ramped up) + uniform-prior regulariser
Optimizer, LR schedule, batch size and the base crop+flip augmentation are identical to the baseline;
the training procedure itself is the method being tested.
Test / validation prediction = average of the two networks' softmax (with flip TTA).
No clean labels are used for anything except measurement in the log.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

import common as C

WARMUP = 10
P_THRESHOLD = 0.5
T_SHARPEN = 0.5
ALPHA = 4.0
LAMBDA_U = 25.0
RAMPUP = 16


@torch.no_grad()
def per_sample_loss(net, D, idx, y_t):
    net.eval()
    losses = []
    idx_t = torch.as_tensor(idx, device=D.device)
    for s in range(0, len(idx), 1000):
        bi = idx_t[s:s + 1000]
        with C.autocast(D.device):
            out = net(D.norm(D.X[bi])).float()
        losses.append(F.cross_entropy(out, y_t[bi], reduction="none"))
    return torch.cat(losses).cpu().numpy()


def gmm_clean_prob(loss):
    loss = (loss - loss.min()) / (loss.max() - loss.min() + 1e-12)
    g = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=C.SEED)
    g.fit(loss.reshape(-1, 1))
    return g.predict_proba(loss.reshape(-1, 1))[:, g.means_.argmin()]


def warmup_epoch(net, opt, scaler, D, tr_t, y_t, ep, epochs):
    net.train()
    perm = tr_t[torch.randperm(len(tr_t), device=D.device)]
    n_iter = len(perm) // C.BATCH_SIZE
    for it in range(n_iter):
        C.cosine_lr(opt, ep + it / n_iter, epochs)
        bi = perm[it * C.BATCH_SIZE:(it + 1) * C.BATCH_SIZE]
        with C.autocast(D.device):
            out = net(D.augment(D.X[bi])).float()
        loss = F.cross_entropy(out, y_t[bi])
        p = F.softmax(out, 1)
        penalty = (p * torch.log(p.clamp_min(1e-8))).sum(1).mean()  # confidence penalty (negative entropy)
        loss = loss + penalty
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


def train_epoch(net, net2, opt, scaler, D, lab_idx, w_lab, unl_idx, y_t, ep, epochs, warm=WARMUP):
    dev = D.device
    net.train()
    net2.eval()
    B = C.BATCH_SIZE
    lab_t = torch.as_tensor(lab_idx, device=dev)
    w_t = torch.as_tensor(w_lab, device=dev, dtype=torch.float32)
    unl_t = torch.as_tensor(unl_idx, device=dev)
    if len(unl_t) == 0:                      # degenerate case: nothing judged noisy
        unl_t = lab_t
    if len(unl_t) < B:                       # tiny unlabelled set: repeat it so a full batch can be drawn
        unl_t = unl_t.repeat((B + len(unl_t) - 1) // len(unl_t))
    lperm = torch.randperm(len(lab_t), device=dev)
    n_iter = len(lab_t) // B
    if n_iter == 0:
        return 0.0
    uperm = torch.randperm(len(unl_t), device=dev)
    up = 0
    prior = torch.ones(C.NUM_CLASSES, device=dev) / C.NUM_CLASSES
    tot = 0.0
    for it in range(n_iter):
        C.cosine_lr(opt, ep + it / n_iter, epochs)
        sel = lperm[it * B:(it + 1) * B]
        bi = lab_t[sel]
        w = w_t[sel][:, None]
        if up + B > len(uperm):
            uperm = torch.randperm(len(unl_t), device=dev)
            up = 0
        ui = unl_t[uperm[up:up + B]]
        up += B
        x1, x2 = D.augment(D.X[bi]), D.augment(D.X[bi])
        u1, u2 = D.augment(D.X[ui]), D.augment(D.X[ui])
        yx = F.one_hot(y_t[bi], C.NUM_CLASSES).float()
        with torch.no_grad(), C.autocast(dev):
            pu = (F.softmax(net(u1).float(), 1) + F.softmax(net(u2).float(), 1)
                  + F.softmax(net2(u1).float(), 1) + F.softmax(net2(u2).float(), 1)) / 4
            ptu = pu ** (1 / T_SHARPEN)
            tu = ptu / ptu.sum(1, keepdim=True)
            px = (F.softmax(net(x1).float(), 1) + F.softmax(net(x2).float(), 1)) / 2
            px = w * yx + (1 - w) * px
            ptx = px ** (1 / T_SHARPEN)
            tx = ptx / ptx.sum(1, keepdim=True)
        lam = np.random.beta(ALPHA, ALPHA)
        lam = max(lam, 1 - lam)
        all_in = torch.cat([x1, x2, u1, u2], 0)
        all_t = torch.cat([tx, tx, tu, tu], 0)
        idx = torch.randperm(all_in.size(0), device=dev)
        mixed_in = lam * all_in + (1 - lam) * all_in[idx]
        mixed_t = lam * all_t + (1 - lam) * all_t[idx]
        with C.autocast(dev):
            logits = net(mixed_in.contiguous(memory_format=torch.channels_last)).float()
        lx = logits[:2 * B]
        lu = logits[2 * B:]
        Lx = -torch.mean(torch.sum(F.log_softmax(lx, 1) * mixed_t[:2 * B], 1))
        Lu = torch.mean((F.softmax(lu, 1) - mixed_t[2 * B:]) ** 2)
        current = ep + it / n_iter
        lamb_u = LAMBDA_U * float(np.clip((current - warm) / RAMPUP, 0.0, 1.0))
        pred_mean = F.softmax(logits, 1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean.clamp_min(1e-8)))
        loss = Lx + lamb_u * Lu + penalty
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if it % 20 == 0:
            tot += loss.item()
    return tot / max(1, (n_iter + 19) // 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", choices=["noisy", "synthetic"], required=True)
    ap.add_argument("--select", choices=["noisy", "synthetic"], required=True)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    assert a.epochs <= 60
    os.makedirs(a.out, exist_ok=True)
    warm = WARMUP if not a.smoke else 1
    C.set_seed(a.seed)
    dev = C.get_device()
    torch.backends.cudnn.benchmark = True
    D = C.Data(a.data, dev, smoke=a.smoke)
    y_all = D.labels[a.labels]
    y_t = torch.as_tensor(y_all, device=dev)
    tr = D.train_idx
    tr_t = torch.as_tensor(tr, device=dev)
    vy_sel = D.labels[a.select][D.val_idx]
    vy = {k: D.labels[k][D.val_idx] for k in ("noisy", "clean", "synthetic")}
    tr_is_clean = (D.labels[a.labels][tr] == D.labels["clean"][tr])   # measurement only

    nets = [C.make_model(dev), C.make_model(dev)]
    opts = [C.make_optimizer(n) for n in nets]
    scalers = [C.make_scaler(dev), C.make_scaler(dev)]
    hist, test_preds = [], []
    best, best_tp = (-1.0, -1), None
    t0 = time.time()
    probs = [None, None]
    for ep in range(a.epochs):
        rec = {"epoch": ep + 1}
        if ep < warm:
            for k in range(2):
                warmup_epoch(nets[k], opts[k], scalers[k], D, tr_t, y_t, ep, a.epochs)
            rec["phase"] = "warmup"
        else:
            rec["phase"] = "dividemix"
            for k in range(2):
                probs[k] = gmm_clean_prob(per_sample_loss(nets[k], D, tr, y_t))
            for k in range(2):
                p = probs[1 - k]          # co-divide: division made by the OTHER network
                lab = p > P_THRESHOLD
                rec[f"net{k + 1}_n_labeled"] = int(lab.sum())
                rec[f"net{k + 1}_labeled_precision"] = float(tr_is_clean[lab].mean() * 100) if lab.any() else 0.0
                rec[f"net{k + 1}_loss"] = train_epoch(nets[k], nets[1 - k], opts[k], scalers[k], D,
                                                      tr[lab], p[lab], tr[~lab], y_t, ep, a.epochs, warm)
        vp = C.predict_probs(nets, D, D.X, D.val_idx)
        tp = C.predict_probs(nets, D, D.Xte)
        test_preds.append(tp.argmax(1).astype(np.int8))
        vpred = vp.argmax(1)
        rec.update({"val_acc_sel": C.acc(vpred, vy_sel), "val_acc_noisy": C.acc(vpred, vy["noisy"]),
                    "val_acc_clean": C.acc(vpred, vy["clean"]), "val_acc_synthetic": C.acc(vpred, vy["synthetic"]),
                    "time_s": round(time.time() - t0, 1)})
        if rec["val_acc_sel"] > best[0]:
            best, best_tp = (rec["val_acc_sel"], ep + 1), tp
        hist.append(rec)
        C.log(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))

    # Task-3 style signals from the final networks (for the noise-score ensemble)
    trp = C.predict_probs(nets, D, D.X, tr, tta=True)
    final_clean_prob = None
    if probs[0] is not None:
        final_clean_prob = (gmm_clean_prob(per_sample_loss(nets[0], D, tr, y_t))
                            + gmm_clean_prob(per_sample_loss(nets[1], D, tr, y_t))) / 2
    np.save(os.path.join(a.out, "train_probs_final.npy"), trp.astype(np.float16))
    np.save(os.path.join(a.out, "train_idx.npy"), tr)
    if final_clean_prob is not None:
        np.save(os.path.join(a.out, "train_gmm_clean_prob.npy"), final_clean_prob.astype(np.float32))
    np.save(os.path.join(a.out, "test_preds_epochs.npy"), np.stack(test_preds))
    np.save(os.path.join(a.out, "test_probs_final.npy"), tp)
    np.save(os.path.join(a.out, "test_probs_best.npy"), best_tp)
    C.save_json({"args": vars(a), "best_epoch": best[1], "best_val_acc_sel": best[0], "final": hist[-1],
                 "best": hist[best[1] - 1], "history": hist, "total_time_s": round(time.time() - t0, 1), "gpu": C.gpu_name(),
                 "hyper": {"warmup": WARMUP, "p_threshold": P_THRESHOLD, "T": T_SHARPEN, "alpha": ALPHA,
                           "lambda_u": LAMBDA_U, "rampup": RAMPUP}},
                os.path.join(a.out, "metrics.json"))
    C.save_json({"ok": True}, os.path.join(a.out, "done.json"))
    C.log(f"done. best epoch {best[1]} val_sel {best[0]:.2f}")


if __name__ == "__main__":
    main()
