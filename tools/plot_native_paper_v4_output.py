"""Plot verified benchmark observations; no synthetic experimental points."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    data = json.loads((directory/"plot_data.json").read_bytes())
    analysis = json.loads((directory/"analysis.json").read_bytes())
    for name in ("comparison.png", "comparison.svg", "plot_provenance.json"):
        if (directory/name).exists():
            raise ValueError("Preserve previous figure: "+name)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.1), constrained_layout=True)
    colors = {"a3t_old": "#6b7280", "a3t_safe": "#008576", "sem_native": "#8152a4"}
    labels = {"a3t_old": "A3-T cũ", "a3t_safe": "A3-T mới", "sem_native": "SEM-RRT*"}
    family_labels = {"mixed_obstacles": "Vật cản\nhỗn hợp", "narrow_passage": "Lối\nhẹp",
                     "ridge_pass": "Sống\nnúi", "rolling_terrain": "Lượn\nsóng", "traversable_hill": "Đồi"}
    groups = collections.defaultdict(list)
    for pair in data["pairs"]:
        if pair["competitor"] == "sem_native" and pair["both_final_safe"]:
            groups[(pair["family"], pair["map_id"])].append(pair)
    ordered = sorted(groups)
    assert len(ordered) == 50 and sum(len(groups[k]) for k in ordered) == 496
    for ax, metric, ylabel, title, note in (
        (axes[0, 0], "cost_terrain_m", "Δ chiều dài bám bề mặt (m)", "A. Đường cuối ngắn hơn trên 50 map", "TB ghép cặp −4,77 m; CI95% [−5,76; −3,77]"),
        (axes[0, 1], "pri", "Δ chỉ số risk PRI", "B. PRI đường cuối cao hơn trên 50 map", "TB ghép cặp +5,22; CI95% [4,89; 5,54]")):
        values = [np.mean([p[metric+"_delta"] for p in groups[key]]) for key in ordered]
        ax.axhline(0, color="#333333", linewidth=.9)
        ax.scatter(np.arange(50), values, s=25, color=colors["a3t_safe"], zorder=3)
        for boundary in (9.5, 19.5, 29.5, 39.5):
            ax.axvline(boundary, color="#d5d9de", linewidth=.8)
        ax.set_xticks([4.5, 14.5, 24.5, 34.5, 44.5], [family_labels[k] for k in sorted(family_labels)])
        ax.set_xlim(-1, 50)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.text(.98, .03, note,
                transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.88))
        ax.grid(axis="y", alpha=.2)
    for ax, clock, title, unit, factor in (
        (axes[1, 0], "first_safe_s", "C. Đạt nghiệm an toàn đầu: chỉ pha solve", "ms từ khi bắt đầu solve (thang log)", 1000),
        (axes[1, 1], "first_safe_latency_s", "D. Đạt nghiệm an toàn đầu: gồm khởi tạo", "s từ khi khởi chạy đến callback", 1)):
        for method in ("a3t_old", "a3t_safe", "sem_native"):
            rows = [r for r in data["times"] if r["backend"] == method]
            values = sorted(r[clock]*factor for r in rows if r["attained_safe"] and r[clock] is not None)
            start, end = (.1, 20000) if factor == 1000 else (0., 4.5)
            x = [start, *values, end]
            y = [0., *[(i+1)/500*100 for i in range(len(values))], len(values)/500*100]
            ax.step(x, y, where="post", label=labels[method]+f" ({len(values)}/500)",
                    color=colors[method], linewidth=1.7)
        if factor == 1000:
            ax.set_xscale("log")
            ax.set_xlim(.1, 20000)
            ax.set_xticks([.1, 1, 10, 100, 1000, 20000], ["0,1", "1", "10", "100", "1.000", "20.000"])
        else:
            ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 104)
        ax.set_xlabel(unit)
        ax.set_ylabel("Lượt từng đạt nghiệm an toàn (%)")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.legend(fontsize=9, loc="lower right", frameon=False)
        ax.grid(alpha=.18)
    figure.suptitle("A3-T mới: đường ngắn hơn, PRI cao hơn, lợi thế tốc độ nằm ở pha solve", fontsize=14, fontweight="bold")
    figure.supxlabel("A–B: hiệu A3-T mới − SEM-RRT*, trung bình theo map trên 496 cặp hợp lệ. C–D: mẫu số giữ đủ 500 lượt mỗi thuật toán.", fontsize=9)
    figure.savefig(directory/"comparison.png", dpi=160)
    figure.savefig(directory/"comparison.svg")
    plt.close(figure)
    provenance = dict(python=sys.version, matplotlib=matplotlib.__version__, numpy=np.__version__,
                      plot_data_sha256=hashlib.sha256((directory/"plot_data.json").read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      observations_only=True, simulated_points=0, paired_maps=len(ordered), paired_trials=496)
    (directory/"plot_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(png=str(directory/"comparison.png"), svg=str(directory/"comparison.svg"))))


if __name__ == "__main__":
    main()
