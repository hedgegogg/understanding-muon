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

The planned model is a two-hidden-layer MLP with width 64. The dataset is scikit-learn's digits: 1797 images of size 8×8, split into 1000 training, 300 validation, and 497 test examples. Pixels are divided by 16. Most comparisons use three training seeds.

AdamW and Lion update the full network. Muon updates the hidden weight matrices, with AdamW for the remaining parameters. Pion will use the same parameter grouping once the paper intended by the instructor is confirmed.

## Experiments

The sequence is: initial runs, learning-rate selection, repeated baseline, matrix updates, batch size at equal epochs, batch size at equal steps, incorrect labels, smaller training sets, weight decay, network width, Pion diagnostics, Pion training, Pion with incorrect labels, and a final repeat with new seeds.

The shared script currently implements only week 1. Completed experiments:

| Week | Question | Visual result |
|---|---|---|
| 1 | Do the three recipes begin learning? | [Loss curves](results/week1/losses.png) |

## Results

Week 1 completed three five-epoch runs at seed 11. Validation loss decreased from 2.305044 to 1.044498 with AdamW, 2.085104 with Lion, and 1.355799 with Muon + AdamW. All updates stayed finite. The measured run times totaled 0.403954 seconds, excluding imports, data loading, and checks. One seed and provisional learning rates do not establish an optimizer ranking.

## References

- [Lion: Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675), with [official code](https://github.com/google/automl/tree/master/lion).
- [Muon official implementation](https://github.com/KellerJordan/Muon).
- [High-pass Pion](https://arxiv.org/abs/2605.19282), with [official code](https://github.com/OPTML-Group/Pion).
- [Spectrum-preserving Pion](https://arxiv.org/abs/2605.12492), with [official code](https://github.com/Sphere-AI-Lab/pion).
- [digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

The two Pion papers describe different algorithms. The experiment notes will identify the one used. `experiment.py` credits the fixed official Lion and Muon revisions used. Their required license texts and notices are retained in [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt). Muon is a CPU float32 port, retaining the official polynomial and matrix scaling.
