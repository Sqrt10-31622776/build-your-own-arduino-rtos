"""TWO_STAGE_RESULTS.md 4 節と付録 C (9200 iteration、7 seed) の表を markdown で標準出力に出す。"""
import re
import numpy as np
from analyze_two_stage import series, first_above

SEEDS = range(1, 8)
MODES = ("joint", "baseline")
LINE = re.compile(r"genesis\s+(\S+)\s+(\S+) \|\s+(\S+)\s+(\S+) \|\s+(\S+)\s+(\S+) \|\s+(\d+) \|\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")


def rows(mode, seed):
    import glob, os
    f = [g for g in (f"results/{mode}_s{seed}{suf}.txt" for suf in ("", "_ext", "_ext2")) if os.path.exists(g)][-1]
    dr, out = None, []
    for l in open(f, errors="ignore"):
        if "nominal robot" in l: dr = "なし"
        elif "randomized robot" in l: dr = "あり"
        m = LINE.match(l)
        if m:
            out.append(dict(dr=dr, cx=float(m[1]), cw=float(m[2]), vx=float(m[3]), wz=float(m[4]), roll=float(m[5]),
                            pitch=float(m[6]), falls=int(m[7]), duty=m[8], diag=m[9], lift=m[10], slip=m[11]))
        elif l.startswith("genesis") and "every robot fell" in l:
            p = l.split()
            out.append(dict(dr=dr, cx=float(p[1]), cw=float(p[2]), vx=None, falls=None))
    return out


def cmd(r): return f"{r['cx']:.2f}, {r['cw']:+.2f}"


print("### 4.1 per seed")
print("| モード | seed | 報酬 @2300 | 報酬 @4600 | 報酬 @9200 | エピソード長 @9200 (上限 1000) | 移動平均 > 20 | 移動平均 > 50 | 行動 std | 報酬曲線 (200 iteration ごとの平均) |")
print("|---|---|---|---|---|---|---|---|---|---|")
S = {}
for m in MODES:
    for s in SEEDS:
        r = series(m, s, "Train/mean_reward"); L = series(m, s, "Train/mean_episode_length"); sd = series(m, s, "Policy/mean_std")
        S[m, s] = r
        curve = " ".join(f"{r[i:i+200].mean():.0f}" for i in range(0, len(r), 200))
        print(f"| {m} | {s} | {r[2200:2300].mean():.1f} | {r[4500:4600].mean():.1f} | {r[-100:].mean():.1f} | {L[-100:].mean():.0f} | {first_above(r, 20)} | {first_above(r, 50)} | {sd[-1]:.3f} | {curve} |")
print("\n| モード | 報酬 @2300 | 報酬 @4600 | 報酬 @9200 | 離陸 (移動平均 > 20): 中央値 | 移動平均 > 50: 中央値 |")
print("|---|---|---|---|---|---|")
for m in MODES:
    f = lambda k: np.array([S[m, s][k - 100:k].mean() if k else S[m, s][-100:].mean() for s in SEEDS])
    a, b, c = f(2300), f(4600), f(0)
    t20 = [first_above(S[m, s], 20) for s in SEEDS]; t50 = [first_above(S[m, s], 50) for s in SEEDS]
    print(f"| {m} | {a.mean():.1f} ± {a.std():.1f} | {b.mean():.1f} ± {b.std():.1f} | **{c.mean():.1f} ± {c.std():.1f}** | {np.median(t20):.0f} | {np.median(t50):.0f} |")

print("\n### 4.2 overall")
print("| モード | DR | 全ロボット転倒した条件 | 転倒回数の合計 | 並進 \\|vx−指令\\| (m/s) | 旋回 \\|wz−指令\\| (rad/s) | roll RMS (°) | pitch RMS (°) |")
print("|---|---|---|---|---|---|---|---|")
for m in MODES:
    for dr in ("なし", "あり", None):
        R = [r for s in SEEDS for r in rows(m, s) if dr is None or r["dr"] == dr]
        ok = [r for r in R if r["vx"] is not None]
        print(f"| {m} | {dr or '合計'} | {len(R)-len(ok)} / {len(R)} | {sum(r['falls'] for r in ok)} | {np.mean([abs(r['vx']-r['cx']) for r in ok]):.4f} (n={len(ok)}) | "
              f"{np.mean([abs(r['wz']-r['cw']) for r in ok]):.3f} | {np.mean([r['roll'] for r in ok]):.2f} | {np.mean([r['pitch'] for r in ok]):.2f} |")

print("\n### 4.3 by command (DR なし/あり・7 seed の平均)")
print("| 指令 (vx, wz) | モード | 実測 vx (m/s) | 実測 wz (rad/s) | \\|vx 誤差\\| | \\|wz 誤差\\| | roll RMS (°) | pitch RMS (°) |")
print("|---|---|---|---|---|---|---|---|")
for c in sorted({cmd(r) for r in rows("joint", 1)}):
    for m in MODES:
        R = [r for s in SEEDS for r in rows(m, s) if cmd(r) == c and r["vx"] is not None]
        print(f"| {c} | {m} | {np.mean([r['vx'] for r in R]):+.3f} | {np.mean([r['wz'] for r in R]):+.3f} | {np.mean([abs(r['vx']-r['cx']) for r in R]):.3f} | "
              f"{np.mean([abs(r['wz']-r['cw']) for r in R]):.3f} | {np.mean([r['roll'] for r in R]):.2f} | {np.mean([r['pitch'] for r in R]):.2f} |")

print("\n### 4.4 turn wz (cmd 0, +0.3) per seed (DR なし / あり)")
print("| モード | " + " | ".join(f"seed {s}" for s in SEEDS) + " | 平均 |")
print("|---|" + "---|" * (len(SEEDS) + 1))
for m in MODES:
    cells, allv = [], []
    for s in SEEDS:
        v = {r["dr"]: r["wz"] for r in rows(m, s) if cmd(r) == "0.00, +0.30"}
        cells.append(f"{v['なし']:+.3f} / {v['あり']:+.3f}"); allv += list(v.values())
    print(f"| {m} | " + " | ".join(cells) + f" | {np.mean(allv):+.3f} |")

print("\n### appendix C")
print("| モード | seed | DR | 指令 | vx | wz | roll | pitch | falls | duty | diag | lift | slip |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for m in MODES:
    for s in SEEDS:
        for dr in ("なし", "あり"):
            for r in sorted([r for r in rows(m, s) if r["dr"] == dr], key=cmd):
                if r["vx"] is None: print(f"| {m} | {s} | {dr} | {cmd(r)} | 全ロボット転倒 | | | | | | | | |")
                else: print(f"| {m} | {s} | {dr} | {cmd(r)} | {r['vx']:+.3f} | {r['wz']:+.3f} | {r['roll']:.2f} | {r['pitch']:.2f} | {r['falls']} | {r['duty']} | {r['diag']} | {r['lift']} | {r['slip']} |")
