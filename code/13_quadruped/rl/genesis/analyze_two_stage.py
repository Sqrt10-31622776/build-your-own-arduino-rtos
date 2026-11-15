"""joint / baseline 6 seed の学習統計を tensorboard から集計する (TWO_STAGE_RESULTS.md 4.1 節の表と同じ指標)。"""
import glob
import os
import re
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "logs/rsl_rl/arduino_quad_velocity"


def load(d, tag):
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def series(mode, seed, tag):
    """base -> _ext -> _ext2 を連結する (延長した run は通算の番号で記録される。そうでなければ足す)。"""
    s = []
    for suf in ("", "_ext", "_ext2"):
        runs = sorted(glob.glob(f"{ROOT}/*_{mode}_s{seed}{suf}"))
        if not runs:
            continue
        e = load(runs[-1], tag)
        off = 0 if (not s or e[0][0] >= len(s) - 5) else len(s)
        s += [(st + off, v) for st, v in e]
    return np.array([v for _, v in s])


def first_above(r, thr, w=50):
    ma = np.convolve(r, np.ones(w) / w, mode="valid")
    i = np.nonzero(ma > thr)[0]
    return int(i[0]) + w - 1 if len(i) else None


def evaluate(mode, seed):
    """4600 iteration の評価 (results/*.txt) を [(DR, 指令, vx, wz, roll, pitch, falls)] にする。全転倒は falls=None。"""
    f = [g for g in (f"results/{mode}_s{seed}{suf}.txt" for suf in ("", "_ext", "_ext2")) if os.path.exists(g)][-1]
    out, dr = [], None
    for l in open(f, errors="ignore"):
        if "nominal robot" in l: dr = False
        elif "randomized robot" in l: dr = True
        m = re.match(r"genesis\s+(\S+)\s+(\S+) \|\s+(\S+)\s+(\S+) \|\s+(\S+)\s+(\S+) \|\s+(\d+) \|", l)
        if m: out.append((dr, float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5]), float(m[6]), int(m[7])))
        elif l.startswith("genesis") and "every robot fell" in l: out.append((dr, *map(float, l.split()[1:3]), None, None, None, None, None))
    return out


def eval_summary(seeds):
    for mode in ("joint", "baseline"):
        rows = [r for s in seeds for r in evaluate(mode, s)]
        ok = [r for r in rows if r[3] is not None]
        print(f"{mode} seeds={list(seeds)}: 全転倒 {len(rows) - len(ok)}/{len(rows)}  "
              f"|vx-cmd| {np.mean([abs(r[3] - r[1]) for r in ok]):.4f}  |wz-cmd| {np.mean([abs(r[4] - r[2]) for r in ok]):.3f}  "
              f"roll {np.mean([r[5] for r in ok]):.2f}  pitch {np.mean([r[6] for r in ok]):.2f}  (n={len(ok)})")


if __name__ == "__main__":
    SEEDS = range(1, 8)
    eval_summary(SEEDS); eval_summary(range(1, 7))
    for m in ("joint", "baseline"):
        R = {s: series(m, s, "Train/mean_reward") for s in SEEDS}
        t = [first_above(R[s], 20) for s in SEEDS]
        f = [R[s][-100:].mean() for s in SEEDS]
        f46 = [R[s][4500:4600].mean() for s in SEEDS]
        print(m, "n_iter", [len(R[s]) for s in SEEDS], "t20", t, "median", np.median([x for x in t if x]), "離陸", sum(x is not None for x in t))
        print("  最終報酬(末尾100) 7seed", f"{np.mean(f):.1f}±{np.std(f):.1f}", "6seed", f"{np.mean(f[:6]):.1f}±{np.std(f[:6]):.1f}", " 4600時点", f"{np.mean(f46):.1f}±{np.std(f46):.1f}", [round(x, 1) for x in f])
    for mode in ("joint", "baseline"):
        for seed in SEEDS:
            r = series(mode, seed, "Train/mean_reward")
            L = series(mode, seed, "Train/mean_episode_length")
            std = series(mode, seed, "Policy/mean_std")
            n = len(r)
            curve = " ".join(f"{r[i:i + 200].mean():.0f}" for i in range(0, n, 200))
            print(f"{mode} s{seed} n={n} len={L[-100:].mean():.0f} t20={first_above(r, 20)} t50={first_above(r, 50)} std={std[-1]:.3f}\n  {curve}")
