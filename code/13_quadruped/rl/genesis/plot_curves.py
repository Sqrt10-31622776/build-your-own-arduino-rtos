"""frozen / oracle_all の学習曲線 (平均報酬、100 iter 移動平均) を results/*.png に描く。"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_two_stage import series, load, ROOT

TAG = "Train/mean_reward"
W = 100


def run(name):
    d = sorted(glob.glob(f"{ROOT}/*_{name}"))[-1]
    return np.array([v for _, v in load(d, TAG)])


def ma(r):
    return np.arange(W - 1, len(r)), np.convolve(r, np.ones(W) / W, mode="valid")


def panel(ax, curves, seed, vline=None):
    for label, r, off, col in curves:
        ax.plot(np.arange(len(r)) + off, r, color=col, alpha=0.15, lw=0.5)
        x, y = ma(r)
        ax.plot(x + off, y, color=col, lw=1.6, label=label)
    if vline:
        ax.axvline(vline, color="gray", ls=":", lw=1)
    ax.set_title(f"seed {seed}")
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)


def fig(path, seeds, build, title, vline=None):
    f, axs = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4), sharey=True)
    for ax, s in zip(axs, seeds):
        panel(ax, build(s), s, vline)
    axs[0].set_ylabel("mean reward")
    axs[0].legend(loc="lower right")
    f.suptitle(title)
    f.tight_layout()
    f.savefig(path, dpi=130)


fig("results/frozen_curves.png", (1, 4, 7),
    lambda s: [("baseline (A only)", series("baseline", s, TAG), 0, "tab:blue"),
               ("frozen (A frozen + B, from 4600)", run(f"frozenA_s{s}"), 4600, "tab:red")],
    "frozen vs baseline (100-iter moving average)", vline=4600)
fig("results/oracle_all_curves.png", (1, 2, 4),
    lambda s: [("baseline", series("baseline", s, TAG), 0, "tab:blue"),
               ("joint", series("joint", s, TAG), 0, "tab:green"),
               ("oracle_all", run(f"oracle_all_s{s}"), 0, "tab:red")],
    "oracle_all vs baseline / joint (100-iter moving average)")
