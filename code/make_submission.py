"""Builds predictions/, a filled results.json draft and analysis numbers/figures from the run outputs.

  python make_submission.py --data DATA_DIR --out RUN_OUT_DIR --sub SUBMISSION_DIR

Clean labels are read here ONLY to measure things (Task 1 confusion matrix, Task 2 curves, the Task 3
self-check AUROCs reported in analysis.json). No choice made here depends on them: the early-stopping epoch
uses noisy validation accuracy, the Task 4 best method uses noisy validation accuracy, the Task 5 checkpoint
uses synthetic validation accuracy, and the Task 3 score formula is fixed in advance.
"""
import argparse
import csv
import os

import numpy as np

import common as C
import noise_tools as NT

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
T4_METHODS = {"gce": "GCE (generalized cross-entropy, q=0.7)",
              "filter": "CV filter + relabel (Task 3 scores) then CE retrain",
              "dividemix": "DivideMix (2-network co-divide + MixMatch)"}


def rank01(x):
    r = np.empty(len(x), np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / (len(x) - 1)


def auroc(score, pos):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pos.astype(int), score))


def prec_at_k(score, pos):
    k = int(pos.sum())
    top = np.argsort(-score, kind="stable")[:k]
    return float(pos[top].mean())


def load_metrics(out, name):
    p = os.path.join(out, name, "metrics.json")
    return C.load_json(p) if os.path.isfile(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--smoke", action="store_true", help="pipeline test on the tiny subset only")
    a = ap.parse_args()
    O = a.out
    P = os.path.join(a.sub, "predictions")
    os.makedirs(P, exist_ok=True)
    fig_dir = os.path.join(O, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    tr = np.loadtxt(os.path.join(a.data, "train_idx.txt"), dtype=np.int64)
    va = np.loadtxt(os.path.join(a.data, "val_idx.txt"), dtype=np.int64)
    noisy, synth, clean = (NT.read_labels(a.data, c) for c in ("noisy", "synthetic", "clean"))
    A = {}

    # ------------------------------------------------------------- Task 1: data
    mis = noisy[tr] != clean[tr]
    A["noise_rate_train_pct"] = float(mis.mean() * 100)
    A["noise_rate_val_pct"] = float((noisy[va] != clean[va]).mean() * 100)
    A["noise_rate_all50k_pct"] = float((noisy != clean).mean() * 100)
    A["synthetic_noise_rate_train_pct"] = float((synth[tr] != clean[tr]).mean() * 100)
    cm = np.zeros((10, 10), np.int64)
    np.add.at(cm, (clean[tr], noisy[tr]), 1)
    cms = np.zeros((10, 10), np.int64)
    np.add.at(cms, (clean[tr], synth[tr]), 1)
    A["confusion_clean_rows_noisy_cols"] = cm.tolist()
    A["confusion_synth_clean_rows"] = cms.tolist()
    with open(os.path.join(O, "t1_confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clean\\noisy"] + CLASSES)
        for i in range(10):
            w.writerow([CLASSES[i]] + cm[i].tolist())
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            pairs.append((int(off[i, j] + off[j, i]), CLASSES[i], CLASSES[j], int(off[i, j]), int(off[j, i])))
    pairs.sort(reverse=True)
    A["top_noise_pairs"] = [{"pair": f"{p[1]}<->{p[2]}", "total": p[0], f"{p[1]}->{p[2]}": p[3], f"{p[2]}->{p[1]}": p[4],
                             "share_of_all_noise_pct": 100 * p[0] / off.sum()} for p in pairs[:8]]
    flows = sorted(((int(off[i, j]), CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10) if i != j), reverse=True)
    A["top_directed_flips"] = [{"clean->noisy": f"{c}->{d}", "count": n} for n, c, d in flows[:10]]
    A["per_class_noise_rate_pct"] = {CLASSES[i]: float(100 * off[i].sum() / cm[i].sum()) for i in range(10)}

    # ------------------------------------------------------------- Task 1: models
    m_clean, m_noisy = load_metrics(O, "t1_clean"), load_metrics(O, "t1_noisy")
    C.write_pred_csv(os.path.join(P, "t1_clean_ref.csv"), np.load(os.path.join(O, "t1_clean", "test_probs_final.npy")).argmax(1))
    C.write_pred_csv(os.path.join(P, "t1_noisy_ce.csv"), np.load(os.path.join(O, "t1_noisy", "test_probs_final.npy")).argmax(1))
    A["t1_clean_final"] = m_clean["final"]
    A["t1_noisy_final"] = m_noisy["final"]
    agree = (np.load(os.path.join(O, "t1_clean", "test_probs_final.npy")).argmax(1) ==
             np.load(os.path.join(O, "t1_noisy", "test_probs_final.npy")).argmax(1)).mean()
    A["t1_test_agreement_clean_vs_noisy_pct"] = float(agree * 100)

    # ------------------------------------------------------------- Task 2
    H = m_noisy["history"]
    with open(os.path.join(P, "t2_dynamics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"])
        for h in H:
            w.writerow([h["epoch"], f"{h['train_acc_correct']:.4f}", f"{h['train_acc_mislabeled']:.4f}",
                        f"{h['val_acc_noisy']:.4f}", f"{h['val_acc_clean']:.4f}"])
    vn = np.array([h["val_acc_noisy"] for h in H])
    es = int(np.argmax(vn)) + 1           # first epoch with the highest NOISY validation accuracy
    tpe = np.load(os.path.join(O, "t1_noisy", "test_preds_epochs.npy"))
    C.write_pred_csv(os.path.join(P, "t2_early_stop.csv"), tpe[es - 1])
    misacc = np.array([h["train_acc_mislabeled"] for h in H])
    vc = np.array([h["val_acc_clean"] for h in H])
    # candidate memorisation onset: after the minimum of train_acc_mislabeled, first epoch from which it stays
    # >= 2 points above that minimum for the rest of training
    emin = int(np.argmin(misacc))
    onset = None
    for e in range(emin, len(misacc)):
        if (misacc[e:] >= misacc[emin] + 2.0).all():
            onset = e + 1
            break
    A["t2"] = {"early_stop_epoch": es, "val_acc_noisy_at_es": float(vn[es - 1]), "val_acc_clean_at_es": float(vc[es - 1]),
               "mislabeled_fit_min_epoch": emin + 1, "mislabeled_fit_min": float(misacc[emin]),
               "memorization_onset_candidate": onset, "val_clean_peak_epoch": int(np.argmax(vc)) + 1,
               "val_clean_peak": float(vc.max()), "val_noisy_peak_epoch": es,
               "final_train_acc_mislabeled": float(misacc[-1]), "final_val_clean": float(vc[-1]),
               "final_val_noisy": float(vn[-1])}

    # ------------------------------------------------------------- Task 3
    cvs = np.load(os.path.join(O, "cvscore_noisy", "score_cv_conf.npy"))
    cv_tr = np.load(os.path.join(O, "cvscore_noisy", "train_idx.npy"))
    oof = np.load(os.path.join(O, "cvscore_noisy", "oof_probs.npy"))
    tr_full = cv_tr
    if a.smoke:
        tr = np.load(os.path.join(O, "dividemix_noisy", "train_idx.npy"))
        pos = np.searchsorted(cv_tr, tr)
        cvs, oof = cvs[pos], oof[pos]
        mis = noisy[tr] != clean[tr]
    assert (cv_tr[np.searchsorted(cv_tr, tr)] == tr).all()
    dm_p = np.load(os.path.join(O, "dividemix_noisy", "train_probs_final.npy")).astype(np.float32)
    dm_tr = np.load(os.path.join(O, "dividemix_noisy", "train_idx.npy"))
    assert (dm_tr == tr).all()
    yl = noisy[tr]
    dm_score = 1.0 - dm_p[np.arange(len(tr)), yl]
    final = 0.5 * rank01(cvs) + 0.5 * rank01(dm_score)      # fixed-in-advance ensemble of two signals
    full_score = dict(zip(tr.tolist(), final.tolist()))
    with open(os.path.join(P, "t3_noise_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "score"])
        for i in tr_full:
            w.writerow([int(i), f"{full_score.get(int(i), 0.0):.6f}"])
    dyn = np.load(os.path.join(O, "t1_noisy", "train_dyn.npz"))
    signals = {"FINAL_submitted (rank-avg CV + DivideMix)": final,
               "CV out-of-fold 1-p(label)": cvs,
               "CV out-of-fold disagreement (argmax != label)": (oof.argmax(1) != yl).astype(float) + 1e-3 * cvs,
               "DivideMix final ensemble 1-p(label)": dm_score,
               "T1 CE loss at early-stop epoch": dyn["loss"][es - 1].astype(np.float32),
               "T1 CE loss at final epoch": dyn["loss"][-1].astype(np.float32),
               "T1 CE mean loss over all epochs": dyn["loss"].astype(np.float32).mean(0),
               "T1 CE negative mean margin (AUM-like)": -dyn["margin"].astype(np.float32).mean(0)}
    gp = os.path.join(O, "dividemix_noisy", "train_gmm_clean_prob.npy")
    if os.path.isfile(gp):
        signals["DivideMix GMM 1-p(clean)"] = 1.0 - np.load(gp)
    A["t3_signals"] = {k: {"auroc": auroc(v, mis), "precision_at_k": prec_at_k(v, mis)} for k, v in signals.items()}
    A["t3_k_true_mislabeled"] = int(mis.sum())
    # failure-mode material: per-class recall of the final score at k
    k = int(mis.sum())
    top = np.zeros(len(tr), bool); top[np.argsort(-final, kind="stable")[:k]] = True
    A["t3_per_true_class"] = {CLASSES[c]: {"recall_of_mislabeled": float(top[mis & (clean[tr] == c)].mean()),
                                           "false_flag_rate_correct": float(top[~mis & (clean[tr] == c)].mean())}
                              for c in range(10)}
    fl = {}
    for (ci, ni) in [(1, 9), (9, 1), (3, 5), (5, 3), (4, 7), (4, 5), (0, 8), (8, 0)]:
        sel = mis & (clean[tr] == ci) & (noisy[tr] == ni)
        fl[f"{CLASSES[ci]}->{CLASSES[ni]}"] = float(top[sel].mean()) if sel.any() else None
    A["t3_recall_by_flip"] = fl
    np.save(os.path.join(O, "t3_final_scores.npy"), final)

    # ------------------------------------------------------------- Task 4
    t4 = []
    for m in ["gce", "filter", "dividemix"]:
        mm = load_metrics(O, f"{m}_noisy")
        if mm is None:
            continue
        fname = f"t4_{m}.csv"
        C.write_pred_csv(os.path.join(P, fname), np.load(os.path.join(O, f"{m}_noisy", "test_probs_best.npy")).argmax(1))
        t4.append({"key": m, "name": m, "description": T4_METHODS[m], "file": fname,
                   "val_acc_noisy_labels": mm["best_val_acc_sel"], "best_epoch": mm["best_epoch"],
                   "val_acc_clean_at_best (measurement only)": mm["best"]["val_acc_clean"],
                   "final_val_noisy": mm["final"]["val_acc_noisy"], "final_val_clean": mm["final"]["val_acc_clean"]})
    best = max(t4, key=lambda r: r["val_acc_noisy_labels"])
    A["t4"] = t4
    A["t4_best"] = best["key"]
    fs = os.path.join(O, "cvscore_noisy", "filter_stats.json")
    if os.path.isfile(fs):
        A["t4_filter_stats"] = C.load_json(fs)
        ov = np.load(os.path.join(O, "cvscore_noisy", "label_override.npy"))
        kept = ov[tr] >= 0
        A["t4_filter_stats"]["label_accuracy_of_kept_set_pct (measurement)"] = float((ov[tr][kept] == clean[tr][kept]).mean() * 100)
    if os.path.isfile(os.path.join(O, "dividemix_noisy", "metrics.json")):
        dh = load_metrics(O, "dividemix_noisy")["history"]
        A["t4_dividemix_division"] = [{k2: h.get(k2) for k2 in ("epoch", "net1_n_labeled", "net1_labeled_precision")}
                                      for h in dh if h.get("phase") == "dividemix"][::5]

    # ------------------------------------------------------------- Task 5
    m5b = load_metrics(O, "t5_ce_synth")
    C.write_pred_csv(os.path.join(P, "t5_baseline_synth.csv"), np.load(os.path.join(O, "t5_ce_synth", "test_probs_final.npy")).argmax(1))
    m5 = load_metrics(O, f"{best['key']}_synth")
    C.write_pred_csv(os.path.join(P, "t5_best_synth.csv"), np.load(os.path.join(O, f"{best['key']}_synth", "test_probs_best.npy")).argmax(1))
    syn = {"ce_final": {"val_synth": m5b["final"]["val_acc_synthetic"], "val_clean": m5b["final"]["val_acc_clean"]},
           "ce_early_stop": {"epoch": m5b["best_epoch"], "val_synth": m5b["best"]["val_acc_synthetic"],
                             "val_clean": m5b["best"]["val_acc_clean"]}}
    for m in ["gce", "filter", "dividemix"]:
        mm = load_metrics(O, f"{m}_synth")
        if mm:
            syn[m] = {"epoch": mm["best_epoch"], "val_synth": mm["best_val_acc_sel"], "val_clean": mm["best"]["val_acc_clean"]}
    real = {"ce_final": {"val_noisy": m_noisy["final"]["val_acc_noisy"], "val_clean": m_noisy["final"]["val_acc_clean"]},
            "ce_early_stop": {"epoch": es, "val_noisy": float(vn[es - 1]), "val_clean": float(vc[es - 1])}}
    for r in t4:
        real[r["key"]] = {"val_noisy": r["val_acc_noisy_labels"], "val_clean": r["val_acc_clean_at_best (measurement only)"]}
    A["t5"] = {"synthetic": syn, "real": real, "best_method": best["key"]}
    if os.path.isfile(os.path.join(O, "cvscore_synth", "filter_stats.json")):
        A["t5_filter_stats_synth"] = C.load_json(os.path.join(O, "cvscore_synth", "filter_stats.json"))
    # synthetic CE dynamics (memorisation is visible through val accuracy)
    A["t5_ce_synth_val_curve"] = [(h["epoch"], h["val_acc_synthetic"], h["val_acc_clean"]) for h in m5b["history"]]

    # ------------------------------------------------------------- results.json draft
    res = {
        "name": "Subodh Shailendra Patel",
        "roll_no": "24B2509",
        "seed": C.SEED,
        "training_setup": {
            "optimizer": "SGD (momentum 0.9, Nesterov, weight decay 5e-4), initial LR 0.1",
            "learning_rate_schedule": "cosine annealing from 0.1 to 0 over the run, updated every iteration",
            "batch_size": C.BATCH_SIZE,
            "epochs": C.EPOCHS,
            "augmentation": "random crop 32x32 with 4-pixel zero padding + random horizontal flip",
            "gpu_used": f"2x NVIDIA {m_noisy.get('gpu', '')} (Kaggle); one network per GPU",
        },
        "task1": {"noise_rate": round(A["noise_rate_train_pct"], 2),
                  "val_acc_noisy_labels__clean_ref_model": round(m_clean["final"]["val_acc_noisy"], 2),
                  "val_acc_noisy_labels__noisy_ce_model": round(m_noisy["final"]["val_acc_noisy"], 2)},
        "task2": {"memorization_start_epoch": onset if onset is not None else A["t2"]["val_clean_peak_epoch"],
                  "early_stop_epoch": es,
                  "val_acc_noisy_labels_at_early_stop": round(float(vn[es - 1]), 2)},
        "task3": {"method": "Rank-average of two signals: (a) 4-fold cross-validated out-of-fold confidence, "
                            "1 - p(noisy label), averaged over each fold model's 5 best noisy-val epochs with flip TTA; "
                            "(b) 1 - p(noisy label) from the final DivideMix two-network ensemble (Task 4)."},
        "task4": {"methods": [{"name": r["name"], "file": r["file"], "val_acc_noisy_labels": round(r["val_acc_noisy_labels"], 2)}
                              for r in t4],
                  "best_method": best["name"]},
        "task5": {"val_acc_synthetic_labels__baseline": round(m5b["final"]["val_acc_synthetic"], 2),
                  "val_acc_synthetic_labels__best_method": round(m5["best_val_acc_sel"], 2)},
    }
    C.save_json(res, os.path.join(a.sub, "results.json"))
    C.save_json(A, os.path.join(O, "analysis.json"))
    make_figures(O, fig_dir, cm, H, es, A)
    print("wrote", P, "and results.json; analysis in", os.path.join(O, "analysis.json"))


def make_figures(O, fig_dir, cm, H, es, A):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    rn = cm / cm.sum(1, keepdims=True) * 100
    im = ax.imshow(rn, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{rn[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if rn[i, j] > 50 else "black")
    ax.set_xlabel("noisy (human) label"); ax.set_ylabel("clean (true) label")
    ax.set_title("CIFAR-10N worst: % of each true class given each noisy label")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "confusion.png"), dpi=150); plt.close(fig)

    e = [h["epoch"] for h in H]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, lab in [("train_acc_correct", "train acc - correctly labelled"), ("train_acc_mislabeled", "train acc - mislabelled (fits wrong label)"),
                     ("val_acc_noisy", "val acc - noisy labels"), ("val_acc_clean", "val acc - clean labels")]:
        ax.plot(e, [h[key] for h in H], label=lab)
    ax.axvline(es, ls="--", c="gray"); ax.text(es, 3, f" early stop ({es})", color="gray")
    on = A["t2"]["memorization_onset_candidate"]
    if on:
        ax.axvline(on, ls=":", c="red"); ax.text(on, 3, f"memorisation ({on}) ", color="red", ha="right")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 101); ax.grid(alpha=.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    ax.set_title("Task 2: training dynamics of CE on noisy labels")
    fig.set_size_inches(8, 5.6); fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "dynamics.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
