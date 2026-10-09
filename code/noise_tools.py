"""Helpers for Task 3 (cross-validated noise scores) and the Task 4 filter/relabel method.

  python noise_tools.py folds   --data D --labels noisy --out OUT/folds_noisy
  python noise_tools.py cvscore --data D --labels noisy --runs OUT/cv_noisy_f0 ... --out OUT/cvscore_noisy
  python noise_tools.py filter  --cvscore OUT/cvscore_noisy --out OUT/cvscore_noisy

Rules respected: only the training label column given by --labels is used; clean labels are never read here.
"""
import argparse
import os

import numpy as np

import common as C

K_FOLDS = 4
CV_EPOCHS = 30
TOP_EPOCHS = 5          # average held-out probabilities over the 5 epochs with best validation accuracy
RELABEL_CONF = 0.9      # Task 4 filter: relabel when the CV ensemble is at least this confident


def read_labels(data_dir, col):
    import csv
    y = np.zeros(50000, np.int64)
    with open(os.path.join(data_dir, "labels.csv")) as f:
        for row in csv.DictReader(f):
            y[int(row["index"])] = int(row[col + "_label"])
    return y


def cmd_folds(a):
    os.makedirs(a.out, exist_ok=True)
    tr = np.loadtxt(os.path.join(a.data, "train_idx.txt"), dtype=np.int64)
    if a.smoke:
        rs = np.random.RandomState(0)
        tr = np.sort(rs.choice(tr, 1200, replace=False))
    y = read_labels(a.data, a.labels)[tr]
    rs = np.random.RandomState(C.SEED)
    fold = np.zeros(len(tr), np.int64)
    for c in range(C.NUM_CLASSES):              # stratified by the (noisy) training label
        ids = np.where(y == c)[0]
        rs.shuffle(ids)
        fold[ids] = np.arange(len(ids)) % K_FOLDS
    # deterministic; written atomically because several CV jobs may call this at the same time
    for k in range(K_FOLDS):
        for name, arr in ((f"f{k}_train.npy", tr[fold != k]), (f"f{k}_held.npy", tr[fold == k])):
            dst = os.path.join(a.out, name)
            if os.path.isfile(dst):
                continue
            tmp = dst + f".tmp{os.getpid()}.npy"
            np.save(tmp, arr)
            os.replace(tmp, dst)
    print("folds:", np.bincount(fold))


def cmd_cvscore(a):
    os.makedirs(a.out, exist_ok=True)
    tr = np.loadtxt(os.path.join(a.data, "train_idx.txt"), dtype=np.int64)
    y = read_labels(a.data, a.labels)
    pos = {int(i): n for n, i in enumerate(tr)}
    oof = np.full((len(tr), C.NUM_CLASSES), np.nan, np.float32)
    used_epochs = {}
    for r in a.runs:
        m = C.load_json(os.path.join(r, "metrics.json"))
        vals = np.array([h["val_acc_sel"] for h in m["history"]])
        top = np.argsort(-vals, kind="stable")[:TOP_EPOCHS]
        hp = np.load(os.path.join(r, "heldout_probs.npy")).astype(np.float32)
        hidx = np.load(os.path.join(r, "heldout_idx.npy"))
        rows = [pos[int(i)] for i in hidx]
        oof[rows] = hp[top].mean(0)
        used_epochs[os.path.basename(r)] = (top + 1).tolist()
    covered = ~np.isnan(oof[:, 0])
    if not covered.all():
        print(f"warning: {(~covered).sum()} training samples without CV prediction (smoke mode?)")
        oof[~covered] = 1.0 / C.NUM_CLASSES
    yl = y[tr]
    p_label = oof[np.arange(len(tr)), yl]
    np.save(os.path.join(a.out, "oof_probs.npy"), oof)
    np.save(os.path.join(a.out, "train_idx.npy"), tr)
    np.save(os.path.join(a.out, "score_cv_conf.npy"), (1.0 - p_label).astype(np.float32))
    C.save_json({"used_epochs": used_epochs, "labels": a.labels}, os.path.join(a.out, "cvscore_info.json"))
    print("cv oof: agreement with training label = %.2f%%" % (100 * (oof.argmax(1) == yl).mean()))


def cmd_filter(a):
    oof = np.load(os.path.join(a.cvscore, "oof_probs.npy"))
    tr = np.load(os.path.join(a.cvscore, "train_idx.npy"))
    info = C.load_json(os.path.join(a.cvscore, "cvscore_info.json"))
    y = read_labels(a.data, info["labels"])
    yl = y[tr]
    pred, conf = oof.argmax(1), oof.max(1)
    new = np.full(50000, -1, np.int64)                       # -1 = drop
    keep = pred == yl                                         # CV ensemble agrees with the given label -> keep
    relabel = (~keep) & (conf >= RELABEL_CONF)                # confident disagreement -> relabel
    new[tr[keep]] = yl[keep]
    new[tr[relabel]] = pred[relabel]
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "label_override.npy"), new)
    stats = {"kept": int(keep.sum()), "relabeled": int(relabel.sum()), "dropped": int(len(tr) - keep.sum() - relabel.sum())}
    C.save_json(stats, os.path.join(a.out, "filter_stats.json"))
    print("filter:", stats)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("folds")
    f.add_argument("--data", required=True); f.add_argument("--labels", required=True)
    f.add_argument("--out", required=True); f.add_argument("--smoke", action="store_true")
    s = sp.add_parser("cvscore")
    s.add_argument("--data", required=True); s.add_argument("--labels", required=True)
    s.add_argument("--runs", nargs="+", required=True); s.add_argument("--out", required=True)
    g = sp.add_parser("filter")
    g.add_argument("--data", required=True); g.add_argument("--cvscore", required=True); g.add_argument("--out", required=True)
    a = ap.parse_args()
    {"folds": cmd_folds, "cvscore": cmd_cvscore, "filter": cmd_filter}[a.cmd](a)


if __name__ == "__main__":
    main()
