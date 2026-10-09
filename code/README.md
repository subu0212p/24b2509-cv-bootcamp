# CIFAR-10N (worst) – learning from noisy human labels

All code is plain PyTorch. The model is the provided `model.py` ResNet-18 (unchanged, trained from scratch).
Runs were done on Kaggle (2 × NVIDIA T4). Seed 42 everywhere. Everything runs from the folder containing these scripts.

## Shared training setup (identical for the baseline and every method)
- SGD, momentum 0.9, Nesterov, weight decay 5e-4, initial LR 0.1, cosine schedule to 0 (per iteration)
- batch size 128, 60 epochs per network (Task 3 CV folds: 30 epochs)
- augmentation: random crop 32×32 with 4-px zero padding + random horizontal flip (done on the GPU)
- fp16 mixed precision; validation/test predictions average the image and its horizontal flip

## Files
| file | purpose |
|---|---|
| `common.py` | data loading, GPU augmentation, optimiser/schedule, evaluation |
| `train.py` | one network with CE or GCE; options for label column, CV subsets, relabel/filter, Task 2 tracking |
| `dividemix.py` | DivideMix (two networks) |
| `noise_tools.py` | CV folds, out-of-fold noise scores, filter/relabel table |
| `run_all.py` | runs every job (two at a time on two GPUs), resumable |
| `make_submission.py` | writes `predictions/*.csv`, `results.json`, analysis numbers and figures |
| `kaggle_phaseA.ipynb`, `kaggle_phaseB.ipynb` | the two Kaggle notebooks that were run |

## Reproduce everything
`DATA` = folder with the provided `train_images.npy, test_images.npy, labels.csv, train_idx.txt, val_idx.txt`.
```bash
python run_all.py --data DATA --out runs --phase A     # Tasks 1-4 + Task 5 baseline   (1 h 37 min on 2xT4)
python run_all.py --data DATA --out runs --phase B     # Task 5 best method (+ the other methods on synthetic labels for Q5)  (1 h 28 min)
python make_submission.py --data DATA --out runs --sub submission
```
(`--phase AB` runs both in one go. A job folder containing `done.json` is skipped, so an interrupted run can be restarted.)

## What each task runs (individual commands, as launched by `run_all.py`)
**Task 1** – reference and noisy baselines (final epoch is submitted)
```bash
python train.py --data DATA --labels clean --out runs/t1_clean
python train.py --data DATA --labels noisy --track_train --out runs/t1_noisy
```
Noise rate + 10×10 confusion matrix: computed in `make_submission.py` (`runs/t1_confusion_matrix.csv`, `runs/figures/confusion.png`).

**Task 2** – `runs/t1_noisy` logs, after every epoch, training accuracy (w.r.t. the noisy label) on correctly-labelled and
mislabelled training images and validation accuracy with noisy and clean labels. The early-stopping epoch is the epoch with the
highest **noisy** validation accuracy; its test predictions (saved every epoch) are written to `t2_early_stop.csv`.

**Task 3** – 4-fold cross-validation on the training split (4 runs, 30 epochs each), stratified by noisy label:
```bash
python noise_tools.py folds --data DATA --labels noisy --out runs/folds_noisy
python train.py --data DATA --labels noisy --epochs 30 --train_subset runs/folds_noisy/fK_train.npy \
       --heldout runs/folds_noisy/fK_held.npy --out runs/cv_noisy_fK            # K = 0..3
python noise_tools.py cvscore --data DATA --labels noisy --runs runs/cv_noisy_f0 runs/cv_noisy_f1 runs/cv_noisy_f2 runs/cv_noisy_f3 --out runs/cvscore_noisy
```
Out-of-fold probability of the noisy label = average over the 5 epochs with best noisy-validation accuracy (flip TTA).
Final score (fixed in advance) = average of the ranks of (a) `1 - p_oof(noisy label)` and (b) `1 - p(noisy label)` from the
final DivideMix ensemble of Task 4. Clean labels are used only to measure AUROC / precision@k afterwards.

**Task 4** – three methods, same budget/setup as the Task 1 noisy baseline; checkpoint chosen by noisy validation accuracy
```bash
python train.py --data DATA --labels noisy --loss gce --out runs/gce_noisy                          # robust loss (GCE, q=0.7)
python noise_tools.py filter --data DATA --cvscore runs/cvscore_noisy --out runs/cvscore_noisy       # keep / relabel / drop
python train.py --data DATA --labels noisy --label_override runs/cvscore_noisy/label_override.npy --out runs/filter_noisy
python dividemix.py --data DATA --labels noisy --select noisy --out runs/dividemix_noisy           # co-training, 2 networks
```
Filter rule: keep a sample if the out-of-fold prediction equals its noisy label; relabel it to the predicted class if the
out-of-fold confidence is ≥ 0.9; otherwise drop it.

**Task 5** – synthetic (symmetric) labels; model selection with synthetic validation labels
```bash
python train.py --data DATA --labels synthetic --select synthetic --out runs/t5_ce_synth            # CE baseline, final epoch
# best Task 4 method by noisy-val accuracy = CV filter + relabel, redone from scratch on the synthetic labels:
python noise_tools.py folds --data DATA --labels synthetic --out runs/folds_synth
python train.py --data DATA --labels synthetic --select synthetic --epochs 30 --train_subset runs/folds_synth/fK_train.npy \
       --heldout runs/folds_synth/fK_held.npy --out runs/cv_synth_fK                                   # K = 0..3
python noise_tools.py cvscore --data DATA --labels synthetic --runs runs/cv_synth_f0 runs/cv_synth_f1 runs/cv_synth_f2 runs/cv_synth_f3 --out runs/cvscore_synth
python noise_tools.py filter --data DATA --cvscore runs/cvscore_synth --out runs/cvscore_synth
python train.py --data DATA --labels synthetic --select synthetic --label_override runs/cvscore_synth/label_override.npy --out runs/filter_synth
```
`t5_baseline_synth.csv` = final epoch of `t5_ce_synth`; `t5_best_synth.csv` = best synthetic-val epoch of `filter_synth`.
The best method is read automatically from `runs/task4_best.json`. GCE and DivideMix were also run on synthetic labels
(`--labels synthetic --select synthetic`) only for the Q5 comparison; they are not submitted.

**Submission files**
```bash
python make_submission.py --data DATA --out runs --sub submission
python check_format.py submission
```
Note: `--smoke` flags exist only to test the pipeline on CPU with a tiny subset; they were not used for the submitted runs.
