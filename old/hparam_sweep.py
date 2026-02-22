"""
hparam_sweep.py
───────────────
1. Reads sweep_results.csv.
2. Keeps only pairs where gap_closed_pct > 0.
3. For each such pair:
   a. Trains the teacher once (standard CE).
   b. Sweeps temperature T  (alpha fixed at 0.5).
   c. Sweeps alpha          (T fixed at 3.0).
4. Saves raw numbers to hparam_results.csv.
5. Saves one plot per pair + a combined overview → visualizations/hparam_*.png
"""

import ast
import csv
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 10
BATCH_SIZE = 256
LR         = 1e-3
SEED       = 42

# Scales to sweep
TEMPERATURE_SCALE = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
ALPHA_SCALE       = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

# Fixed defaults while sweeping the *other* param
DEFAULT_T     = 3.0
DEFAULT_ALPHA = 0.5

OUT_DIR = "visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers, in_size = [], 784
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers.append(nn.Linear(in_size, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# ── Helpers ───────────────────────────────────────────────────────────────────
def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            correct += (model(imgs).argmax(1) == lbls).sum().item()
            total   += lbls.size(0)
    model.train()
    return correct / total


def train_standard(model):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(EPOCHS):
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(imgs), lbls).backward()
            opt.step()
        print(f"    epoch {epoch+1}/{EPOCHS}  acc={evaluate(model):.4f}")
    return evaluate(model)


def train_distill(student, teacher, T, alpha):
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    teacher.eval()
    student.train()
    for epoch in range(EPOCHS):
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(imgs)
            opt.zero_grad()
            s_logits = student(imgs)
            soft = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction="batchmean",
            ) * T ** 2
            hard = F.cross_entropy(s_logits, lbls)
            (alpha * soft + (1 - alpha) * hard).backward()
            opt.step()
        print(f"    epoch {epoch+1}/{EPOCHS}  acc={evaluate(student):.4f}")
    return evaluate(student)


def parse_cfg(s):
    return ast.literal_eval(s)

# ── Load positive pairs ───────────────────────────────────────────────────────
sweep_df = pd.read_csv("sweep_results.csv")
positive = sweep_df[sweep_df["gap_closed_pct"] > 0].copy()
positive["teacher_cfg_parsed"] = positive["teacher_cfg"].apply(parse_cfg)
positive["student_cfg_parsed"] = positive["student_cfg"].apply(parse_cfg)

print(f"Found {len(positive)} positive pairs:")
for _, r in positive.iterrows():
    print(f"  T={r['teacher_cfg']}  S={r['student_cfg']}  gap={r['gap_closed_pct']:.1f}%")

# ── Main sweep ────────────────────────────────────────────────────────────────
all_results = []

for _, row in positive.iterrows():
    t_cfg = row["teacher_cfg_parsed"]
    s_cfg = row["student_cfg_parsed"]
    t_label = str(t_cfg)
    s_label = str(s_cfg)
    baseline_acc = row["baseline_acc"]

    print(f"\n{'='*60}")
    print(f" Pair  Teacher {t_label}  →  Student {s_label}")
    print(f"{'='*60}")

    # Train teacher once for this pair
    print("\n-- Training teacher --")
    teacher = MLP(t_cfg).to(DEVICE)
    teacher_acc = train_standard(teacher)
    print(f"  Teacher acc: {teacher_acc:.4f}")

    # ── Temperature sweep (alpha fixed) ──────────────────────────────────────
    temp_results = []
    print(f"\n-- Temperature sweep (alpha={DEFAULT_ALPHA}) --")
    for T in TEMPERATURE_SCALE:
        print(f"  T={T}")
        student = MLP(s_cfg).to(DEVICE)
        acc = train_distill(student, teacher, T=T, alpha=DEFAULT_ALPHA)
        print(f"  → acc={acc:.4f}")
        temp_results.append({"T": T, "acc": acc})
        all_results.append({
            "teacher_cfg": t_label, "student_cfg": s_label,
            "sweep": "temperature", "param_name": "T", "param_value": T,
            "alpha": DEFAULT_ALPHA, "acc": round(acc, 4),
            "baseline_acc": baseline_acc, "teacher_acc": teacher_acc,
        })

    # ── Alpha sweep (T fixed) ─────────────────────────────────────────────────
    alpha_results = []
    print(f"\n-- Alpha sweep (T={DEFAULT_T}) --")
    for alpha in ALPHA_SCALE:
        print(f"  alpha={alpha}")
        student = MLP(s_cfg).to(DEVICE)
        acc = train_distill(student, teacher, T=DEFAULT_T, alpha=alpha)
        print(f"  → acc={acc:.4f}")
        alpha_results.append({"alpha": alpha, "acc": acc})
        all_results.append({
            "teacher_cfg": t_label, "student_cfg": s_label,
            "sweep": "alpha", "param_name": "alpha", "param_value": alpha,
            "T": DEFAULT_T, "acc": round(acc, 4),
            "baseline_acc": baseline_acc, "teacher_acc": teacher_acc,
        })

    # ── Per-pair plot ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Teacher {t_label}  →  Student {s_label}", fontsize=11, fontweight="bold")

    # Temperature
    ts  = [r["T"]   for r in temp_results]
    acs = [r["acc"] for r in temp_results]
    ax1.plot(ts, acs, marker="o", color="#2176ae", zorder=3)
    ax1.axhline(baseline_acc, ls="--", color="gray", label=f"baseline ({baseline_acc:.4f})")
    ax1.axhline(teacher_acc,  ls=":",  color="green", label=f"teacher ({teacher_acc:.4f})")
    ax1.set_xlabel("Temperature (T)")
    ax1.set_ylabel("Test accuracy")
    ax1.set_title(f"Temperature sweep  (alpha={DEFAULT_ALPHA})")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    best_t = temp_results[int(np.argmax(acs))]
    ax1.annotate(f"best T={best_t['T']}\n{best_t['acc']:.4f}",
                 (best_t["T"], best_t["acc"]),
                 xytext=(8, -18), textcoords="offset points", fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8))

    # Alpha
    als  = [r["alpha"] for r in alpha_results]
    acas = [r["acc"]   for r in alpha_results]
    ax2.plot(als, acas, marker="o", color="#e76f51", zorder=3)
    ax2.axhline(baseline_acc, ls="--", color="gray", label=f"baseline ({baseline_acc:.4f})")
    ax2.axhline(teacher_acc,  ls=":",  color="green", label=f"teacher ({teacher_acc:.4f})")
    ax2.set_xlabel("Alpha (weight on soft/KD loss)")
    ax2.set_ylabel("Test accuracy")
    ax2.set_title(f"Alpha sweep  (T={DEFAULT_T})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    best_a = alpha_results[int(np.argmax(acas))]
    ax2.annotate(f"best α={best_a['alpha']}\n{best_a['acc']:.4f}",
                 (best_a["alpha"], best_a["acc"]),
                 xytext=(8, -18), textcoords="offset points", fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8))

    safe_name = f"{t_label}_{s_label}".replace(" ", "").replace(",", "_")
    path = f"{OUT_DIR}/hparam_{safe_name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = "hparam_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)
print(f"\nRaw results saved to {csv_path}")

# ── Combined overview plot ────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
pairs = results_df[["teacher_cfg", "student_cfg"]].drop_duplicates().values.tolist()
n = len(pairs)

fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)
fig.suptitle("Hparam sweep – all positive pairs", fontsize=13, fontweight="bold", y=1.01)

for i, (t_label, s_label) in enumerate(pairs):
    sub = results_df[(results_df["teacher_cfg"] == t_label) & (results_df["student_cfg"] == s_label)]
    baseline_acc = sub["baseline_acc"].iloc[0]
    teacher_acc  = sub["teacher_acc"].iloc[0]

    # temperature subplot
    t_sub = sub[sub["sweep"] == "temperature"].sort_values("param_value")
    axes[i, 0].plot(t_sub["param_value"], t_sub["acc"], marker="o", color="#2176ae")
    axes[i, 0].axhline(baseline_acc, ls="--", color="gray", lw=0.9)
    axes[i, 0].axhline(teacher_acc,  ls=":",  color="green", lw=0.9)
    axes[i, 0].set_title(f"T sweep | {t_label} → {s_label}")
    axes[i, 0].set_xlabel("Temperature")
    axes[i, 0].set_ylabel("Accuracy")
    axes[i, 0].grid(alpha=0.3)

    # alpha subplot
    a_sub = sub[sub["sweep"] == "alpha"].sort_values("param_value")
    axes[i, 1].plot(a_sub["param_value"], a_sub["acc"], marker="o", color="#e76f51")
    axes[i, 1].axhline(baseline_acc, ls="--", color="gray", lw=0.9, label="baseline")
    axes[i, 1].axhline(teacher_acc,  ls=":",  color="green", lw=0.9, label="teacher")
    axes[i, 1].set_title(f"Alpha sweep | {t_label} → {s_label}")
    axes[i, 1].set_xlabel("Alpha")
    axes[i, 1].set_ylabel("Accuracy")
    axes[i, 1].grid(alpha=0.3)
    axes[i, 1].legend(fontsize=7)

plt.tight_layout()
overview_path = f"{OUT_DIR}/hparam_overview.png"
plt.savefig(overview_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Overview plot saved to {overview_path}")
