"""
sweep.py  –  grid search over teacher and student sizes for knowledge distillation.

For every (teacher_sizes, student_sizes) pair we:
  1. Train the teacher with standard cross-entropy.
  2. Train a baseline student with standard cross-entropy (no distillation).
  3. Train a distilled student using the teacher's soft targets.

Results are printed as a table and saved to sweep_results.csv.
"""

import csv
import itertools

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
TEMPERATURE = 3.0
ALPHA       = 0.5   # weight on soft (KD) loss
SEED        = 42

torch.manual_seed(SEED)

# Sizes to sweep — each entry is a hidden-layer spec (list of ints).
TEACHER_CONFIGS = [
    [256],
    [512, 512],
    [1024, 512, 256],
]

STUDENT_CONFIGS = [
    [32],
    [64],
    [128, 64],
]

# ── Data ──────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        in_size = 784
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

# ── Losses ────────────────────────────────────────────────────────────────────
def distillation_loss(s_logits, t_logits, labels, T=TEMPERATURE, alpha=ALPHA):
    soft = F.kl_div(
        F.log_softmax(s_logits / T, dim=1),
        F.softmax(t_logits / T, dim=1),
        reduction='batchmean',
    ) * T ** 2
    hard = F.cross_entropy(s_logits, labels)
    return alpha * soft + (1 - alpha) * hard

# ── Training / evaluation ─────────────────────────────────────────────────────
def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            correct += (model(images).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    model.train()
    return correct / total


def train_standard(model, label, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(epochs):
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(train_loader)
        acc = evaluate(model)
        print(f"  [{label}] epoch {epoch+1:02d}/{epochs}  loss={avg:.4f}  acc={acc:.4f}")
    return evaluate(model)


def train_distill(student, teacher, label, epochs=EPOCHS):
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    teacher.eval()
    student.train()
    for epoch in range(epochs):
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(images)
            opt.zero_grad()
            loss = distillation_loss(student(images), t_logits, labels)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(train_loader)
        acc = evaluate(student)
        print(f"  [{label}] epoch {epoch+1:02d}/{epochs}  loss={avg:.4f}  acc={acc:.4f}")
    return evaluate(student)

# ── Sweep ─────────────────────────────────────────────────────────────────────
results = []

for t_cfg, s_cfg in itertools.product(TEACHER_CONFIGS, STUDENT_CONFIGS):
    t_label = f"T{t_cfg}"
    s_label = f"S{s_cfg}"
    print(f"\n{'='*60}")
    print(f" Teacher {t_label}  →  Student {s_label}")
    print(f"{'='*60}")

    # 1. Teacher
    print(f"\n-- Training teacher {t_label} --")
    teacher = MLP(t_cfg).to(DEVICE)
    teacher_acc = train_standard(teacher, label=t_label)
    print(f"  Teacher final acc: {teacher_acc:.4f}  ({teacher.param_count():,} params)")

    # 2. Baseline student (same arch as distilled, no KD)
    print(f"\n-- Training baseline student {s_label} (no distillation) --")
    baseline = MLP(s_cfg).to(DEVICE)
    baseline_acc = train_standard(baseline, label=f"{s_label}-base")
    print(f"  Baseline final acc: {baseline_acc:.4f}  ({baseline.param_count():,} params)")

    # 3. Distilled student
    print(f"\n-- Training distilled student {s_label} --")
    distilled = MLP(s_cfg).to(DEVICE)
    distilled_acc = train_distill(distilled, teacher, label=f"{s_label}-kd")
    print(f"  Distilled final acc: {distilled_acc:.4f}")

    gap = teacher_acc - baseline_acc
    recovered = (distilled_acc - baseline_acc) / gap * 100 if gap > 1e-6 else float('nan')

    results.append({
        'teacher_cfg'   : str(t_cfg),
        'student_cfg'   : str(s_cfg),
        'teacher_params': teacher.param_count(),
        'student_params': baseline.param_count(),
        'teacher_acc'   : round(teacher_acc,   4),
        'baseline_acc'  : round(baseline_acc,  4),
        'distilled_acc' : round(distilled_acc, 4),
        'gap_closed_pct': round(recovered,     1),
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(" SWEEP RESULTS")
print(f"{'='*60}")
header = f"{'Teacher':20s} {'Student':15s} {'T_acc':>6} {'Base':>6} {'Dist':>6} {'Gap%':>6}"
print(header)
print('-' * len(header))
for r in results:
    print(
        f"{r['teacher_cfg']:20s} {r['student_cfg']:15s} "
        f"{r['teacher_acc']:6.4f} {r['baseline_acc']:6.4f} "
        f"{r['distilled_acc']:6.4f} {r['gap_closed_pct']:5.1f}%"
    )

# Save CSV
csv_path = "sweep_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nResults saved to {csv_path}")
