"""
visualize.py  –  generate plots and a summary table from sweep_results.csv.
All outputs are saved to ./visualizations/
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

OUT_DIR = "visualizations"

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("sweep_results.csv")

# Short labels for axes
df["teacher_label"] = df["teacher_cfg"].str.replace(" ", "")
df["student_label"] = df["student_cfg"].str.replace(" ", "")

teachers = df["teacher_label"].unique()
students = df["student_label"].unique()

# ── 1. Grouped bar chart: baseline vs distilled per student, grouped by teacher ──
fig, ax = plt.subplots(figsize=(10, 5))

n_teachers = len(teachers)
n_students = len(students)
x = np.arange(n_teachers)
bar_w = 0.12
offsets = np.linspace(-(n_students - 1) * bar_w, (n_students - 1) * bar_w, n_students * 2)

colors_base = ["#a8c4e0", "#f4a261", "#81b29a"]
colors_dist = ["#2176ae", "#e76f51", "#3a7d44"]

for i, student in enumerate(students):
    sub = df[df["student_label"] == student].set_index("teacher_label").loc[teachers]
    ax.bar(x + offsets[i * 2],     sub["baseline_acc"],  width=bar_w, color=colors_base[i],
           label=f"{student} baseline", zorder=2)
    ax.bar(x + offsets[i * 2 + 1], sub["distilled_acc"], width=bar_w, color=colors_dist[i],
           label=f"{student} distilled", zorder=2)

ax.set_xticks(x)
ax.set_xticklabels(teachers, fontsize=9)
ax.set_xlabel("Teacher architecture")
ax.set_ylabel("Test accuracy")
ax.set_title("Baseline vs Distilled accuracy per student, grouped by teacher")
ax.set_ylim(0.955, 0.985)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.3f}"))
ax.legend(fontsize=7, ncol=3, loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/1_grouped_bar.png", dpi=150)
plt.close()
print("Saved 1_grouped_bar.png")

# ── 2. Heatmaps: baseline / distilled / gap_closed ────────────────────────────
def make_heatmap(matrix, row_labels, col_labels, title, fmt, cmap, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Student architecture"); ax.set_ylabel("Teacher architecture")
    ax.set_title(title)
    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            val = matrix[r, c]
            text = "nan" if math.isnan(val) else fmt.format(val)
            ax.text(c, r, text, ha="center", va="center", fontsize=10,
                    color="white" if abs(val - (vmin or 0)) > (((vmax or 1) - (vmin or 0)) * 0.6) else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

pivot_base = df.pivot(index="teacher_label", columns="student_label", values="baseline_acc").loc[teachers, students].values.astype(float)
pivot_dist = df.pivot(index="teacher_label", columns="student_label", values="distilled_acc").loc[teachers, students].values.astype(float)
pivot_gap  = df.pivot(index="teacher_label", columns="student_label", values="gap_closed_pct").loc[teachers, students].values.astype(float)

make_heatmap(pivot_base, teachers, students,
             "Baseline student accuracy", "{:.4f}", "Blues",
             f"{OUT_DIR}/2_heatmap_baseline.png", vmin=0.96, vmax=0.98)

make_heatmap(pivot_dist, teachers, students,
             "Distilled student accuracy", "{:.4f}", "Greens",
             f"{OUT_DIR}/3_heatmap_distilled.png", vmin=0.96, vmax=0.98)

make_heatmap(pivot_gap, teachers, students,
             "Gap closed by distillation (%)", "{:.1f}", "RdYlGn",
             f"{OUT_DIR}/4_heatmap_gap_closed.png", vmin=-100, vmax=100)

# ── 3. Scatter: param ratio vs accuracy gain ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
df["param_ratio"] = df["teacher_params"] / df["student_params"]
df["acc_gain"]    = df["distilled_acc"] - df["baseline_acc"]

sc = ax.scatter(df["param_ratio"], df["acc_gain"] * 100,
                c=df["student_params"], cmap="plasma", s=90, zorder=3)
plt.colorbar(sc, ax=ax, label="Student param count")

for _, row in df.iterrows():
    ax.annotate(f"{row['teacher_label']}\n→{row['student_label']}",
                (row["param_ratio"], row["acc_gain"] * 100),
                fontsize=6, xytext=(4, 4), textcoords="offset points")

ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Teacher / student param ratio")
ax.set_ylabel("Accuracy gain from distillation (pp)")
ax.set_title("Does a larger teacher-to-student ratio help?")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/5_scatter_ratio_vs_gain.png", dpi=150)
plt.close()
print("Saved 5_scatter_ratio_vs_gain.png")

# ── 4. Summary table image ────────────────────────────────────────────────────
table_df = df[["teacher_label", "student_label",
               "teacher_acc", "baseline_acc", "distilled_acc", "gap_closed_pct"]].copy()
table_df.columns = ["Teacher", "Student", "Teacher acc", "Baseline acc", "Distilled acc", "Gap closed %"]
table_df = table_df.round(4)

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.axis("off")
tbl = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

# Colour header row
for j in range(len(table_df.columns)):
    tbl[0, j].set_facecolor("#2176ae")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Highlight rows where distillation helps
for i, (_, row) in enumerate(table_df.iterrows(), start=1):
    gain = row["Distilled acc"] - row["Baseline acc"]
    color = "#d4edda" if gain > 0 else "#f8d7da"
    for j in range(len(table_df.columns)):
        tbl[i, j].set_facecolor(color)

plt.title("Sweep results summary (green = distillation helped)", pad=12, fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/6_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 6_summary_table.png")

print(f"\nAll done — check ./{OUT_DIR}/")
