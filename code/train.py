"""Single-network training: cross-entropy or GCE, on clean / noisy / synthetic labels.

Used for: Task 1 (clean ref + noisy CE, with Task 2 dynamics tracking), Task 3 cross-validation folds,
Task 4 GCE and CV-filter/relabel methods, Task 5 CE baseline on synthetic labels.

Outputs in --out:
  metrics.json          per-epoch log + summary (best epoch by the selection labels)
  test_preds_epochs.npy (E,10000) int8   test predictions after every epoch
  test_probs_final.npy / test_probs_best.npy  (10000,10)
  [--track_train]  train_dyn.npz  per-epoch, per-sample loss / margin / prediction on the training set
  [--heldout]      heldout_probs.npy (E, n_heldout, 10) float16  + heldout_idx.npy
  done.json
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C


def gce_loss(logits, y, q=0.7):
    p = F.softmax(logits.float(), 1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
    return ((1.0 - p ** q) / q).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", choices=["clean", "noisy", "synthetic"], required=True,
                    help="which label column the network is trained on")
    ap.add_argument("--select", choices=["noisy", "synthetic"], default="noisy",
                    help="validation labels used for model selection")
    ap.add_argument("--loss", choices=["ce", "gce"], default="ce")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--train_subset", default=None, help=".npy of training indices to use (default: all of train_idx)")
    ap.add_argument("--heldout", default=None, help=".npy of training indices to predict every epoch (CV)")
    ap.add_argument("--label_override", default=None,
                    help=".npy int array over 50k: new training label, -1 = drop the sample (Task 4 filter)")
    ap.add_argument("--track_train", action="store_true", help="Task 2 dynamics: evaluate the whole train split every epoch")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    assert a.epochs <= 60
    os.makedirs(a.out, exist_ok=True)

    C.set_seed(a.seed)
    dev = C.get_device()
    torch.backends.cudnn.benchmark = True
    D = C.Data(a.data, dev, smoke=a.smoke)

    y_all = D.labels[a.labels].copy()
    tr = D.train_idx if a.train_subset is None else np.load(a.train_subset).astype(np.int64)
    assert np.isin(tr, D.train_idx).all(), "training indices must come from train_idx"
    assert not np.isin(tr, D.val_idx).any()
    if a.label_override is not None:
        ov = np.load(a.label_override).astype(np.int64)
        y_all = np.where(ov >= 0, ov, y_all)
        tr = tr[ov[tr] >= 0]
    C.log(f"train samples: {len(tr)}  labels={a.labels} loss={a.loss}")
    y_t = torch.as_tensor(y_all, device=dev)
    tr_t = torch.as_tensor(tr, device=dev)

    held = None if a.heldout is None else np.load(a.heldout).astype(np.int64)
    if held is not None:
        assert not np.isin(held, tr).any()

    net = C.make_model(dev)
    opt = C.make_optimizer(net)
    scaler = C.make_scaler(dev)
    n_iter = len(tr) // C.BATCH_SIZE  # drop last incomplete batch

    vy_sel = D.labels[a.select][D.val_idx]
    vy_noisy, vy_clean, vy_syn = (D.labels[k][D.val_idx] for k in ("noisy", "clean", "synthetic"))
    trk_noisy = D.labels[a.labels][D.train_idx]          # labels the network is trained on
    trk_clean = D.labels["clean"][D.train_idx]          # measurement only
    correct_mask = trk_noisy == trk_clean

    hist, test_preds, held_probs = [], [], []
    dyn_loss, dyn_margin, dyn_pred = [], [], []
    best = (-1.0, -1)
    best_test_probs = None
    t0 = time.time()
    lr = C.BASE_LR
    for ep in range(a.epochs):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        tot, nb = 0.0, 0
        for it in range(n_iter):
            lr = C.cosine_lr(opt, ep + it / n_iter, a.epochs)
            bi = perm[it * C.BATCH_SIZE:(it + 1) * C.BATCH_SIZE]
            x = D.augment(D.X[bi])
            y = y_t[bi]
            with C.autocast(dev):
                out = net(x)
                loss = F.cross_entropy(out.float(), y) if a.loss == "ce" else gce_loss(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() if it % 20 == 0 else 0.0
            nb += 1 if it % 20 == 0 else 0

        # ---- evaluation (no augmentation; flip-TTA for val/test)
        vp = C.predict_probs(net, D, D.X, D.val_idx)
        vpred = vp.argmax(1)
        tp = C.predict_probs(net, D, D.Xte)
        test_preds.append(tp.argmax(1).astype(np.int8))
        rec = {"epoch": ep + 1, "lr_end": lr, "train_loss": tot / max(nb, 1),
               "val_acc_sel": C.acc(vpred, vy_sel),
               "val_acc_noisy": C.acc(vpred, vy_noisy), "val_acc_clean": C.acc(vpred, vy_clean),
               "val_acc_synthetic": C.acc(vpred, vy_syn), "time_s": round(time.time() - t0, 1)}
        if a.track_train:
            pp = C.predict_probs(net, D, D.X, D.train_idx, tta=False)
            py = pp[np.arange(len(pp)), trk_noisy]
            other = pp.copy()
            other[np.arange(len(pp)), trk_noisy] = -1
            pred = pp.argmax(1)
            dyn_loss.append((-np.log(np.clip(py, 1e-7, 1))).astype(np.float16))
            dyn_margin.append((py - other.max(1)).astype(np.float16))
            dyn_pred.append(pred.astype(np.int8))
            rec["train_acc_correct"] = C.acc(pred[correct_mask], trk_noisy[correct_mask])
            rec["train_acc_mislabeled"] = C.acc(pred[~correct_mask], trk_noisy[~correct_mask])
            rec["train_mislabeled_pred_clean"] = C.acc(pred[~correct_mask], trk_clean[~correct_mask])
            rec["train_acc_all_given"] = C.acc(pred, trk_noisy)
        if held is not None:
            held_probs.append(C.predict_probs(net, D, D.X, held).astype(np.float16))
        if rec["val_acc_sel"] > best[0]:
            best = (rec["val_acc_sel"], ep + 1)
            best_test_probs = tp
        hist.append(rec)
        C.log(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))

    np.save(os.path.join(a.out, "test_preds_epochs.npy"), np.stack(test_preds))
    np.save(os.path.join(a.out, "test_probs_final.npy"), tp)
    np.save(os.path.join(a.out, "test_probs_best.npy"), best_test_probs)
    if a.track_train:
        np.savez_compressed(os.path.join(a.out, "train_dyn.npz"), loss=np.stack(dyn_loss),
                            margin=np.stack(dyn_margin), pred=np.stack(dyn_pred), train_idx=D.train_idx)
    if held is not None:
        np.save(os.path.join(a.out, "heldout_probs.npy"), np.stack(held_probs))
        np.save(os.path.join(a.out, "heldout_idx.npy"), held)
    summary = {"args": vars(a), "n_train": int(len(tr)), "best_epoch": best[1], "best_val_acc_sel": best[0],
               "final": hist[-1], "best": hist[best[1] - 1], "history": hist,
               "total_time_s": round(time.time() - t0, 1), "gpu": C.gpu_name()}
    C.save_json(summary, os.path.join(a.out, "metrics.json"))
    C.save_json({"ok": True}, os.path.join(a.out, "done.json"))
    C.log(f"done. best epoch {best[1]} val_sel {best[0]:.2f}; final val_sel {hist[-1]['val_acc_sel']:.2f}")


if __name__ == "__main__":
    main()
