# Small experiments with AdamW, Lion, Muon, and Pion

This is a small course project about optimizer behavior. I plan to train a small MLP on digits and check how the results change with learning rate, batch size, incorrect labels, training-set size, and network width. A few matrix experiments will help explain what the updates do.

The focus is on small experiments that can run on a CPU. Results from this setup do not establish which optimizer is best for large models.

## Setup

Tested with Python 3.10.12 on CPU, using PyTorch 2.14.1+cpu, NumPy 2.2.6, and scikit-learn 1.7.2. With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python experiment.py --week 1
```

If `.venv` already exists, skip its creation. Week 1 uses one PyTorch thread and float32. The command trains three models and appends measurements to `.local/results.csv` and epoch evaluations to `.local/curves.csv`. Repeating it trains three new runs. The fixed split is kept in `.local/data.npz`. These local files, caches, and working instructions are ignored by Git.

Visual results are saved in `results/week<week_number>/`. Week 1 produces [training and validation loss curves](results/week1/losses.png). To rebuild the plot from the local CSV files without training:

```bash
.venv/bin/python experiment.py --week 1 --plot-only
```

Run the week-2 calibration with:

```bash
.venv/bin/python experiment.py --week 2
```

This runs 27 new eight-epoch models. Repeating it appends 27 new attempts. Use `--week 2 --plot-only` to rebuild its plot from saved measurements without training. Weekly notes remain local and ignored.

Run the week-3 baseline after calibration with:

```bash
.venv/bin/python experiment.py --week 3
```

This runs nine new 15-epoch models with frozen rates. Repeating it appends nine new attempts. Use `--week 3 --plot-only` to rebuild its loss plot without training. The script checks the frozen rates against the local completed calibration before training.

The planned model is a two-hidden-layer MLP with width 64. The dataset is scikit-learn's digits: 1797 images of size 8×8, split into 1000 training, 300 validation, and 497 test examples. Pixels are divided by 16. Most comparisons use three training seeds.

AdamW and Lion update the full network. Muon updates the hidden weight matrices, with AdamW for the remaining parameters. Pion will use the same parameter grouping once the paper intended by the instructor is confirmed.

## Experiments

The sequence is: initial runs, learning-rate selection, repeated baseline, matrix updates, batch size at equal epochs, batch size at equal steps, incorrect labels, smaller training sets, weight decay, network width, Pion diagnostics, Pion training, Pion with incorrect labels, and a final repeat with new seeds.

The shared script currently implements weeks 1, 2, and 3. Completed experiments:

| Week | Question | Visual result |
|---|---|---|
| 1 | Do the three recipes begin learning? | [Loss curves](results/week1/losses.png) |
| 2 | How sensitive are the recipes to learning rate? | [Learning-rate plot](results/week2/learning_rates.png) |
| 3 | Is the frozen-rate comparison consistent across seeds? | [Loss means and variability](results/week3/losses.png) |

## Results

Week 1 completed three five-epoch runs at seed 11. Validation loss decreased from 2.305044 to 1.044498 with AdamW, 2.085104 with Lion, and 1.355799 with Muon + AdamW. All updates stayed finite. The measured run times totaled 0.403954 seconds, excluding imports, data loading, and checks. One seed and provisional learning rates do not establish an optimizer ranking.

### Week 2

All 27 eight-epoch runs completed with finite values, using seeds 11, 22, and 33. The measured run times totaled 5.315093 seconds, excluding imports, data loading, checks, and plotting. The primary metric below is final validation cross-entropy loss. Each candidate completed 3/3 planned runs.

| Recipe | LR | Loss, seed 11 | Loss, seed 22 | Loss, seed 33 | Mean ± sample SD | Completed |
|---|---:|---:|---:|---:|---:|---:|
| AdamW | 0.0001 | 2.221741 | 2.219205 | 2.195800 | 2.212249 ± 0.014301 | 3/3 |
| AdamW | 0.001 | 0.484182 | 0.469375 | 0.462586 | 0.472047 ± 0.011043 | 3/3 |
| AdamW | 0.01 | 0.207602 | 0.204815 | 0.168107 | 0.193508 ± 0.022042 | 3/3 |
| Lion | 3e-05 | 2.278812 | 2.274091 | 2.267211 | 2.273371 ± 0.005834 | 3/3 |
| Lion | 0.0003 | 1.827264 | 1.793916 | 1.709938 | 1.777039 ± 0.060456 | 3/3 |
| Lion | 0.003 | 0.371937 | 0.596006 | 0.418047 | 0.461996 ± 0.118323 | 3/3 |
| Muon + AdamW | 0.001 | 0.755740 | 0.770051 | 0.761880 | 0.762557 ± 0.007180 | 3/3 |
| Muon + AdamW | 0.01 | 0.143559 | 0.136369 | 0.158738 | 0.146222 ± 0.011420 | 3/3 |
| Muon + AdamW | 0.1 | 0.396214 | 0.221587 | 0.477612 | 0.365138 ± 0.130811 | 3/3 |

The lowest mean final validation loss selected AdamW **0.01**, Lion **0.003**, and Muon **0.01**. AdamW selection finished before Muon; all Muon runs used auxiliary AdamW **0.01**, independently of the hidden-matrix rate. These rates are frozen in `SELECTED_LRS` in `experiment.py` for later experiments. Exact ties choose the smaller rate, and any non-finite run makes a candidate ineligible.

AdamW and Lion selected the upper grid boundaries, so this search does not establish their optimal rates. Increasing Muon's rate from 0.01 to 0.1 raised mean loss from 0.146222 to 0.365138. Three seeds give only a rough variability check. An eight-epoch calibration may not select the best rates for 15 epochs, and this hybrid recipe comparison does not isolate Muon's matrix operation. Test data remain untouched.

### Week 3

All nine baseline runs completed with finite values: three recipes × seeds 11, 22, and 33, with 15 epochs, 240 updates, and 15,000 examples seen per run. The week-2 rates remained frozen: AdamW 0.01, Lion 0.003, Muon 0.01, and auxiliary AdamW 0.01. No retuning or control reuse occurred. The split, model, batch size, and optimizer formulas stayed unchanged.

| Recipe | Seed | Validation loss | Validation accuracy | Run seconds | Status |
|---|---:|---:|---:|---:|---|
| AdamW | 11 | 0.193060 | 96.00% | 0.353539 | completed |
| AdamW | 22 | 0.141453 | 95.67% | 0.318506 | completed |
| AdamW | 33 | 0.182907 | 95.33% | 0.313234 | completed |
| Lion | 11 | 0.632281 | 96.00% | 0.295859 | completed |
| Lion | 22 | 0.690329 | 96.00% | 0.291239 | completed |
| Lion | 33 | 0.654237 | 95.00% | 0.288435 | completed |
| Muon + AdamW | 11 | 0.090213 | 98.00% | 0.445636 | completed |
| Muon + AdamW | 22 | 0.090313 | 97.67% | 0.454102 | completed |
| Muon + AdamW | 33 | 0.088549 | 97.67% | 0.452141 | completed |

| Recipe | Validation loss, mean ± sample SD | Accuracy mean (%) | Accuracy SD (pp) | Completed |
|---|---:|---:|---:|---:|
| AdamW | 0.172473 ± 0.027340 | 95.67 | 0.33 | 3/3 |
| Lion | 0.658949 ± 0.029310 | 95.67 | 0.58 | 3/3 |
| Muon + AdamW | 0.089692 ± 0.000991 | 97.78 | 0.19 | 3/3 |

These summaries use sample standard deviation across three seeds. Accuracy SD is in percentage points (pp). Paired loss differences subtract AdamW at the same seed; negative values favour the recipe in that row.

| Loss difference versus AdamW | Seed 11 | Seed 22 | Seed 33 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Lion | +0.439220 | +0.548877 | +0.471330 | +0.486476 ± 0.056375 |
| Muon + AdamW | -0.102847 | -0.051140 | -0.094358 | -0.082782 ± 0.027729 |

Muon + AdamW had the lowest validation loss at every matching seed and exceeded AdamW's mean accuracy by 2.11 percentage points. AdamW and Lion both averaged 95.67% accuracy, despite different losses. Every baseline reproduced its corresponding week-2 validation loss at epoch 8; no new calibration runs were required.

Measured run times totaled **3.212690 seconds**: 3.054124 training and 0.124059 evaluation seconds, plus overhead. Imports, data loading, checks, and plotting are excluded. The [loss plot](results/week3/losses.png) shows means and shaded sample standard deviations, not confidence intervals. These results describe frozen recipes on one small task. Three seeds, reused validation data, coarse eight-epoch calibration, and auxiliary AdamW limit broader conclusions. Test data remain untouched.

## References

- [Lion: Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675), with [official code](https://github.com/google/automl/tree/master/lion).
- [Muon official implementation](https://github.com/KellerJordan/Muon).
- [High-pass Pion](https://arxiv.org/abs/2605.19282), with [official code](https://github.com/OPTML-Group/Pion).
- [Spectrum-preserving Pion](https://arxiv.org/abs/2605.12492), with [official code](https://github.com/Sphere-AI-Lab/pion).
- [digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

The two Pion papers describe different algorithms. The experiment notes will identify the one used. `experiment.py` credits the fixed official Lion and Muon revisions used. Their required license texts and notices are retained in [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt). Muon is a CPU float32 port, retaining the official polynomial and matrix scaling.
