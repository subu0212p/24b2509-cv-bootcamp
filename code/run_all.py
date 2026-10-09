"""Runs every training job, two at a time when two GPUs are present.

  python run_all.py --data DATA_DIR --out OUT_DIR --phase A
  python run_all.py --data DATA_DIR --out OUT_DIR --phase B [--prev PHASE_A_OUT_DIR]

Phase A: Task 1 (clean + noisy CE, noisy run logs Task 2 dynamics), Task 3 CV folds, Task 4 methods
         (GCE, CV filter/relabel, DivideMix), Task 5 CE baseline on synthetic labels.
Phase B: Task 5 best method on synthetic labels (best = highest NOISY-validation accuracy in Task 4),
         plus the other two Task 4 methods on synthetic labels (used only for the Q5 analysis).
A job whose folder already contains done.json is skipped, so a run can be resumed.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import common as C
import noise_tools as NT

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
METHODS = ["gce", "filter", "dividemix"]


def gpu_count():
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


class Job:
    def __init__(self, name, cmds, deps=(), cost=1.0):
        self.name, self.cmds, self.deps, self.cost = name, cmds, list(deps), cost


def jobs_for(a, labels, include_t1=False):
    """Jobs for one label column ('noisy' or 'synthetic')."""
    O, D = a.out, a.data
    sel = "noisy" if labels == "noisy" else "synthetic"
    tag = "noisy" if labels == "noisy" else "synth"
    sm = ["--smoke"] if a.smoke else []
    ep = ["--epochs", str(a.epochs)]
    cvep = ["--epochs", str(min(a.epochs, NT.CV_EPOCHS))]
    tr = [PY, os.path.join(HERE, "train.py"), "--data", D]
    nt = [PY, os.path.join(HERE, "noise_tools.py")]
    J = {}
    J[f"dividemix_{tag}"] = Job(f"dividemix_{tag}", [[PY, os.path.join(HERE, "dividemix.py"), "--data", D, "--labels", labels,
                                                      "--select", sel, "--out", f"{O}/dividemix_{tag}"] + ep + sm], cost=6)
    if include_t1:
        J["t1_noisy"] = Job("t1_noisy", [tr + ["--labels", "noisy", "--select", "noisy", "--track_train",
                                               "--out", f"{O}/t1_noisy"] + ep + sm], cost=1.3)
        J["t1_clean"] = Job("t1_clean", [tr + ["--labels", "clean", "--select", "noisy", "--out", f"{O}/t1_clean"] + ep + sm])
    if labels == "synthetic":
        J["t5_ce_synth"] = Job("t5_ce_synth", [tr + ["--labels", "synthetic", "--select", "synthetic",
                                                     "--out", f"{O}/t5_ce_synth"] + ep + sm])
    folds = f"{O}/folds_{tag}"
    fold_cmd = nt + ["folds", "--data", D, "--labels", labels, "--out", folds] + sm
    for k in range(NT.K_FOLDS):
        J[f"cv_{tag}_f{k}"] = Job(f"cv_{tag}_f{k}", [fold_cmd, tr + [
            "--labels", labels, "--select", sel, "--train_subset", f"{folds}/f{k}_train.npy",
            "--heldout", f"{folds}/f{k}_held.npy", "--out", f"{O}/cv_{tag}_f{k}"] + cvep + sm], cost=0.4)
    cvs = f"{O}/cvscore_{tag}"
    J[f"filter_{tag}"] = Job(f"filter_{tag}", [
        nt + ["cvscore", "--data", D, "--labels", labels, "--out", cvs, "--runs"] + [f"{O}/cv_{tag}_f{k}" for k in range(NT.K_FOLDS)],
        nt + ["filter", "--data", D, "--cvscore", cvs, "--out", cvs],
        tr + ["--labels", labels, "--select", sel, "--label_override", f"{cvs}/label_override.npy",
              "--out", f"{O}/filter_{tag}"] + ep + sm], deps=[f"cv_{tag}_f{k}" for k in range(NT.K_FOLDS)])
    J[f"gce_{tag}"] = Job(f"gce_{tag}", [tr + ["--labels", labels, "--select", sel, "--loss", "gce",
                                               "--out", f"{O}/gce_{tag}"] + ep + sm])
    return J


def done(a, name):
    return os.path.isfile(os.path.join(a.out, name, "done.json"))


def run_jobs(a, jobs, order):
    ngpu = gpu_count()
    C.log(f"{ngpu} GPU slot(s); jobs: {order}")
    running = {}  # slot -> (job, Popen, cmd_index, logfile)
    pending = [j for j in order if not done(a, j)]
    for j in order:
        if done(a, j):
            C.log(f"skip {j} (already done)")
    failed = []

    def start(slot, job, ci):
        os.makedirs(os.path.join(a.out, job.name), exist_ok=True)
        lf = open(os.path.join(a.out, job.name, f"log_{ci}.txt"), "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot), PYTHONUNBUFFERED="1")
        p = subprocess.Popen(job.cmds[ci], stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        running[slot] = (job, p, ci, lf)
        C.log(f"[gpu{slot}] start {job.name} step {ci + 1}/{len(job.cmds)}")

    while pending or running:
        for slot in range(ngpu):
            if slot in running:
                job, p, ci, lf = running[slot]
                rc = p.poll()
                if rc is None:
                    continue
                lf.close()
                del running[slot]
                if rc != 0:
                    C.log(f"[gpu{slot}] FAILED {job.name} (rc={rc}); see {job.name}/log_{ci}.txt")
                    with open(os.path.join(a.out, job.name, f"log_{ci}.txt")) as f:
                        print("".join(f.readlines()[-30:]))
                    failed.append(job.name)
                    continue
                if ci + 1 < len(job.cmds):
                    start(slot, job, ci + 1)
                    continue
                C.log(f"[gpu{slot}] finished {job.name}")
            # free slot: pick first pending job whose deps are done
            for name in list(pending):
                job = jobs[name]
                if any(d in failed for d in job.deps):
                    pending.remove(name)
                    failed.append(name)
                    C.log(f"cannot run {name}: dependency failed")
                    continue
                if all(done(a, d) for d in job.deps):
                    pending.remove(name)
                    start(slot, job, 0)
                    break
        if not running and pending and not any(all(done(a, d) for d in jobs[n].deps) for n in pending):
            C.log(f"stuck: {pending}")
            failed += pending
            break
        time.sleep(5)
    return failed


def best_method(a):
    res = {}
    for m in METHODS:
        p = os.path.join(a.out, f"{m}_noisy", "metrics.json")
        if os.path.isfile(p):
            res[m] = C.load_json(p)["best_val_acc_sel"]
    best = max(res, key=res.get)
    return best, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["A", "B", "AB"], required=True)
    ap.add_argument("--prev", default=None, help="folder with outputs of an earlier run to resume from")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.prev and os.path.abspath(a.prev) != os.path.abspath(a.out):
        for name in os.listdir(a.prev):
            src, dst = os.path.join(a.prev, name), os.path.join(a.out, name)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
        C.log(f"copied previous outputs from {a.prev}")

    failed = []
    if "A" in a.phase:
        J = jobs_for(a, "noisy", include_t1=True)
        J.update({k: v for k, v in jobs_for(a, "synthetic").items() if k == "t5_ce_synth"})
        order = ["dividemix_noisy", "t1_noisy", "t1_clean"] + [f"cv_noisy_f{k}" for k in range(NT.K_FOLDS)] + \
                ["gce_noisy", "filter_noisy", "t5_ce_synth"]
        failed += run_jobs(a, J, order)
        C.save_json({"failed": failed}, os.path.join(a.out, "phaseA_status.json"))
    if "B" in a.phase:
        best, res = best_method(a)
        C.log(f"Task 4 noisy-val accuracies: {res} -> best = {best}")
        C.save_json({"best_method": best, "noisy_val_acc": res}, os.path.join(a.out, "task4_best.json"))
        J = jobs_for(a, "synthetic")
        # best method's synthetic run first; the other two are for the Q5 comparison only
        order = {"dividemix": ["dividemix_synth"], "filter": [f"cv_synth_f{k}" for k in range(NT.K_FOLDS)] + ["filter_synth"],
                 "gce": ["gce_synth"]}[best]
        rest = ["dividemix_synth", "gce_synth"] + [f"cv_synth_f{k}" for k in range(NT.K_FOLDS)] + ["filter_synth"]
        if "t5_ce_synth" not in os.listdir(a.out) or not done(a, "t5_ce_synth"):
            order = ["t5_ce_synth"] + order
        order += [j for j in rest if j not in order]
        failed += run_jobs(a, J, order)
        C.save_json({"failed": failed}, os.path.join(a.out, "phaseB_status.json"))
    C.log(f"ALL DONE. failed jobs: {failed}")


if __name__ == "__main__":
    main()
