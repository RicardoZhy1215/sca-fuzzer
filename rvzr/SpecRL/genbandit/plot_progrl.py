"""
Reward curve of a progrl run as a standalone SVG (no plotting library needed): mean reward
(violations per program) per update, its rolling mean with a 95% band, the Revizor-distribution
baseline for the same config, and the policy's KL to the prior underneath (separate panel).

  python rvzr/SpecRL/genbandit/plot_progrl.py rvzr/SpecRL/genbandit/runs/<run> [--baseline 21.1]
"""
import argparse
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--baseline", type=float, default=21.1,
                help="violations per 1000 programs under Revizor's distribution, same config")
ap.add_argument("--window", type=int, default=20, help="rolling window, in updates")
args = ap.parse_args()
U = [json.loads(l) for l in open(os.path.join(args.run, "agent.jsonl"))]
x = [u["results"] / 1000 for u in U]                     # thousand programs
r = [1000 * u["r_mean"] for u in U]                      # violations per 1000 programs
n = [u["n"] for u in U]
kl = [u.get("kl", 0.0) for u in U]
W = args.window
sm, lo, hi = [], [], []
for i in range(len(U)):
    a, b = max(0, i - W + 1), i + 1          # trailing window: the last W updates
    v = sum(U[j]["r_mean"] * n[j] for j in range(a, b))
    m = sum(n[a:b])
    p = v / m
    se = math.sqrt(max(p * (1 - p), 1e-12) / m)
    sm.append(1000 * p)
    lo.append(1000 * max(p - 1.96 * se, 0))
    hi.append(1000 * (p + 1.96 * se))

# ---- layout
Wd, Hd = 900, 640
L, R = 78, 150
P1 = (66, 410)        # reward panel top, bottom
P2 = (478, 580)       # KL panel
xmax = max(x) * 1.0
xs = lambda v: L + (Wd - L - R) * v / xmax
ymax1 = 60.0
y1 = lambda v: P1[1] - (P1[1] - P1[0]) * min(v, ymax1) / ymax1
ymax2 = max(0.07, max(kl) * 1.15)
y2 = lambda v: P2[1] - (P2[1] - P2[0]) * v / ymax2
C = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984",
     "grid": "#e6e5e0", "s1": "#2a78d6", "raw": "#a9c8ee"}
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{Hd}" viewBox="0 0 {Wd} {Hd}" '
     f'font-family="Noto Sans CJK SC, Noto Serif CJK SC, sans-serif">',
     f'<rect width="{Wd}" height="{Hd}" fill="{C["surface"]}"/>',
     f'<text x="{L}" y="24" font-size="16" font-weight="600" fill="{C["text"]}">'
     f'奖励曲线：每 1000 个程序中出 violation 的个数（progrl, v4, 12 小时）</text>']
# grids + y labels
for v in range(0, 61, 10):
    o.append(f'<line x1="{L}" x2="{Wd - R}" y1="{y1(v):.1f}" y2="{y1(v):.1f}" stroke="{C["grid"]}" stroke-width="1"/>')
    o.append(f'<text x="{L - 8}" y="{y1(v) + 4:.1f}" font-size="12" text-anchor="end" fill="{C["text2"]}">{v}</text>')
for v in [0, 0.02, 0.04, 0.06]:
    if v <= ymax2:
        o.append(f'<line x1="{L}" x2="{Wd - R}" y1="{y2(v):.1f}" y2="{y2(v):.1f}" stroke="{C["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{L - 8}" y="{y2(v) + 4:.1f}" font-size="12" text-anchor="end" fill="{C["text2"]}">{v:.2f}</text>')
for v in range(0, int(xmax) + 1, 20):
    for (t, b) in (P1, P2):
        o.append(f'<line x1="{xs(v):.1f}" x2="{xs(v):.1f}" y1="{b}" y2="{b + 5}" stroke="{C["muted"]}" stroke-width="1"/>')
    o.append(f'<text x="{xs(v):.1f}" y="{P2[1] + 20}" font-size="12" text-anchor="middle" fill="{C["text2"]}">{v}k</text>')
o.append(f'<text x="{(L + Wd - R) / 2:.0f}" y="{P2[1] + 44}" font-size="13" text-anchor="middle" fill="{C["text2"]}">已训练的程序数（千个）</text>')
o.append(f'<text x="{L}" y="{P1[0] - 10}" font-size="12" fill="{C["text2"]}">violation / 1000 个程序（轴上限 60，个别批次超出）</text>')
o.append(f'<text x="{L}" y="{P2[0] - 8}" font-size="12" fill="{C["text2"]}">策略与 Revizor 先验的 KL 距离（每个决策，nats）</text>')
# raw per-update points
for xi, ri in zip(x, r):
    o.append(f'<circle cx="{xs(xi):.1f}" cy="{y1(ri):.1f}" r="2" fill="{C["raw"]}"><title>{xi:.1f}k 个程序：这一批 {ri:.1f}</title></circle>')
# band + smoothed line
band = " ".join(f"{xs(a):.1f},{y1(b):.1f}" for a, b in zip(x, hi)) + " " + \
    " ".join(f"{xs(a):.1f},{y1(b):.1f}" for a, b in zip(reversed(x), reversed(lo)))
o.append(f'<polygon points="{band}" fill="{C["s1"]}" fill-opacity="0.16"/>')
o.append(f'<polyline points="{" ".join(f"{xs(a):.1f},{y1(b):.1f}" for a, b in zip(x, sm))}" fill="none" stroke="{C["s1"]}" stroke-width="2"/>')
# baseline
yb = y1(args.baseline)
o.append(f'<line x1="{L}" x2="{Wd - R}" y1="{yb:.1f}" y2="{yb:.1f}" stroke="{C["text2"]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
o.append(f'<text x="{Wd - R + 8}" y="{yb + 4:.1f}" font-size="12" fill="{C["text2"]}">Revizor 原生分布</text>')
o.append(f'<text x="{Wd - R + 8}" y="{yb + 19:.1f}" font-size="12" fill="{C["text2"]}">同配置 {args.baseline:.1f}</text>')
o.append(f'<text x="{Wd - R + 8}" y="{y1(sm[-1]) - 4:.1f}" font-size="12" fill="{C["text"]}">滑动平均 {sm[-1]:.1f}</text>')
o.append(f'<text x="{Wd - R + 8}" y="{y1(sm[-1]) + 11:.1f}" font-size="11" fill="{C["text2"]}">（前 {W} 次更新≈{W * 256 // 1000}k 程序，</text>')
o.append(f'<text x="{Wd - R + 8}" y="{y1(sm[-1]) + 25:.1f}" font-size="11" fill="{C["text2"]}">阴影 = 95% 区间）</text>')
o.append(f'<text x="{Wd - R + 8}" y="{P1[0] + 40:.1f}" font-size="11" fill="{C["text2"]}">浅色点 = 单次更新</text>')
o.append(f'<text x="{Wd - R + 8}" y="{P1[0] + 54:.1f}" font-size="11" fill="{C["text2"]}">（256 个程序）</text>')
# KL panel
o.append(f'<polyline points="{" ".join(f"{xs(a):.1f},{y2(b):.1f}" for a, b in zip(x, kl))}" fill="none" stroke="{C["s1"]}" stroke-width="2"/>')
o.append(f'<text x="{Wd - R + 8}" y="{y2(kl[-1]) + 4:.1f}" font-size="12" fill="{C["text"]}">KL {kl[-1]:.3f}</text>')
for (t, b) in (P1, P2):
    o.append(f'<line x1="{L}" x2="{Wd - R}" y1="{b}" y2="{b}" stroke="{C["muted"]}" stroke-width="1"/>')
o.append("</svg>")
out = os.path.join(args.run, "reward_curve.svg")
open(out, "w").write("\n".join(o))
print(out)
