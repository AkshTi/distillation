import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "temperature_sweep.csv"
OUT_PATH = "temperature_sweep_table.png"

# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------
rows = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        rows.append(r)

EPOCHS = max(int(r["epoch"]) for r in rows)

# Final-epoch rows only
final = [r for r in rows if int(r["epoch"]) == EPOCHS]

# Separate baseline (T=0) and distilled runs
baseline_row = next(r for r in final if r["run"] == "baseline")
baseline_acc = float(baseline_row["test_acc"])

temps = sorted(set(int(r["T"]) for r in final if r["run"] == "distilled"))
distilled = {int(r["T"]): r for r in final if r["run"] == "distilled"}

# ---------------------------------------------------------------------------
# Build table data
# Columns: T | train_loss | train_acc | test_acc | Δ vs baseline
# Rows: baseline + one per temperature
# ---------------------------------------------------------------------------
col_labels = ["T", "train_loss", "train_acc", "test_acc", "Δ test_acc"]

def fmt_row(r, T_label):
    acc = float(r["test_acc"])
    delta = acc - baseline_acc
    sign  = "+" if delta >= 0 else ""
    return [
        T_label,
        f"{float(r['train_loss']):.4f}",
        f"{float(r['train_acc']):.4f}",
        f"{acc:.4f}",
        f"{sign}{delta:.4f}",
    ]

table_data = [fmt_row(baseline_row, "baseline\n(T=—)")]
for T in temps:
    table_data.append(fmt_row(distilled[T], str(T)))

# ---------------------------------------------------------------------------
# Colour-map the Δ column (green = better, red = worse)
# ---------------------------------------------------------------------------
n_rows = len(table_data)
n_cols = len(col_labels)

cell_colours = [["#f9f9f9"] * n_cols for _ in range(n_rows)]

deltas = []
for i, row in enumerate(table_data):
    try:
        deltas.append(float(row[-1]))
    except ValueError:
        deltas.append(0.0)

max_abs = max(abs(d) for d in deltas) or 1.0
for i, d in enumerate(deltas):
    if d > 0:
        intensity = 0.3 + 0.5 * (d / max_abs)
        cell_colours[i][-1] = plt.cm.Greens(intensity)
    elif d < 0:
        intensity = 0.3 + 0.5 * (abs(d) / max_abs)
        cell_colours[i][-1] = plt.cm.Reds(intensity)
    else:
        cell_colours[i][-1] = "#eeeeee"

# Highlight best test_acc row
test_accs = [float(r["test_acc"]) for r in [baseline_row] + [distilled[T] for T in temps]]
best_idx  = int(np.argmax(test_accs))
cell_colours[best_idx][3] = "#ffe066"   # gold highlight on best test_acc

# ---------------------------------------------------------------------------
# Draw figure
# ---------------------------------------------------------------------------
fig_h = 0.45 * (n_rows + 2)
fig, ax = plt.subplots(figsize=(9, max(fig_h, 3)))
ax.axis("off")

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellColours=cell_colours,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 1.8)

# Style header
for col in range(n_cols):
    tbl[(0, col)].set_facecolor("#2c3e50")
    tbl[(0, col)].set_text_props(color="white", fontweight="bold")

fig.suptitle(
    f"Temperature Sweep  (alpha=0.1, epochs={EPOCHS})\n"
    f"Baseline test_acc = {baseline_acc:.4f}  |  gold = best test_acc",
    fontsize=12, fontweight="bold", y=0.97
)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
