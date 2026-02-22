import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 256
LR          = 1e-3
SEED        = 42

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# Configurable MLP
# hidden_dims controls model capacity, e.g.:
#   STRONG_DIMS = [512, 512]   (teacher)
#   WEAK_DIMS   = [128, 128]   (student)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dims: list = [512, 512], num_classes: int = 10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


def train_standard(model, loader, optimizer, epochs: int):
    """Train with hard labels; returns list of (epoch, train_loss, train_acc)."""
    model.train()
    history = []
    for epoch in range(1, epochs + 1):
        total_loss, correct = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        train_loss = total_loss / len(loader.dataset)
        train_acc  = correct   / len(loader.dataset)
        test_acc   = evaluate(model, test_loader)
        history.append((epoch, train_loss, train_acc, test_acc))
        print(f"  [Standard] Epoch {epoch:>2} | loss {train_loss:.4f} | train_acc {train_acc:.4f} | test_acc {test_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
# Distillation loss  (Hinton et al. 2015)
#
# L = alpha * CE(student_logits, hard_labels)
#   + (1 - alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T))
# ---------------------------------------------------------------------------
def distillation_loss(student_logits, teacher_logits, labels, T: float, alpha: float):
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * ce_loss + (1.0 - alpha) * kd_loss


def train_distillation(student, teacher, loader, optimizer,
                       epochs: int, T: float, alpha: float):
    """Train student via distillation; returns list of (epoch, train_loss, train_acc, test_acc)."""
    teacher.eval()
    student.train()
    history = []
    for epoch in range(1, epochs + 1):
        total_loss, correct = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                teacher_logits = teacher(x)
            optimizer.zero_grad()
            student_logits = student(x)
            loss = distillation_loss(student_logits, teacher_logits, y, T, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (student_logits.argmax(1) == y).sum().item()
        train_loss = total_loss / len(loader.dataset)
        train_acc  = correct   / len(loader.dataset)
        test_acc   = evaluate(student, test_loader)
        history.append((epoch, train_loss, train_acc, test_acc))
        print(f"  [T={T:.0f}] Epoch {epoch:>2} | loss {train_loss:.4f} | train_acc {train_acc:.4f} | test_acc {test_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)

    TEACHER_EPOCHS = 10
    STUDENT_EPOCHS = 10
    ALPHA          = 0.1   # weight on hard-label CE vs soft KD loss

    STRONG_DIMS = [512, 512]   # teacher
    WEAK_DIMS   = [128, 128]   # baseline student

    TEMPERATURES  = list(range(1, 11))   # T = 1, 2, ..., 10
    T_BEST        = 8                    # fixed T for the model-size sweep

    # Student sizes swept in the second experiment (T fixed at T_BEST)
    STUDENT_SIZES = [
        [32,  32],
        [64,  64],
        [128, 128],
        [256, 256],
    ]

    # --- Train teacher once (used by both sweeps) ---
    print("=== Training teacher (strong model) ===")
    teacher = MLP(hidden_dims=STRONG_DIMS).to(DEVICE)
    opt_t   = torch.optim.Adam(teacher.parameters(), lr=LR)
    train_standard(teacher, train_loader, opt_t, TEACHER_EPOCHS)
    teacher_acc = evaluate(teacher, test_loader)
    print(f"  Teacher test acc: {teacher_acc:.4f}\n")

    # --- Train baseline student once ---
    print("=== Training student from scratch (baseline) ===")
    torch.manual_seed(SEED)
    student_baseline = MLP(hidden_dims=WEAK_DIMS).to(DEVICE)
    opt_sb = torch.optim.Adam(student_baseline.parameters(), lr=LR)
    baseline_history = train_standard(student_baseline, train_loader, opt_sb, STUDENT_EPOCHS)
    baseline_acc = evaluate(student_baseline, test_loader)
    print(f"  Student (baseline) test acc: {baseline_acc:.4f}\n")

    # =========================================================================
    # Experiment 1 — Temperature sweep  (student fixed at WEAK_DIMS)
    # =========================================================================
    temp_rows = []

    for (epoch, train_loss, train_acc, test_acc) in baseline_history:
        temp_rows.append({
            "run":        "baseline",
            "student_dims": str(WEAK_DIMS),
            "T":          0,
            "alpha":      "-",
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "test_acc":   round(test_acc,   6),
        })

    for T in TEMPERATURES:
        print(f"=== [Temp sweep] Distilling  T={T}  student={WEAK_DIMS} ===")
        torch.manual_seed(SEED)
        student_kd = MLP(hidden_dims=WEAK_DIMS).to(DEVICE)
        opt_kd     = torch.optim.Adam(student_kd.parameters(), lr=LR)
        history    = train_distillation(student_kd, teacher, train_loader, opt_kd,
                                        STUDENT_EPOCHS, T=T, alpha=ALPHA)
        for (epoch, train_loss, train_acc, test_acc) in history:
            temp_rows.append({
                "run":          "distilled",
                "student_dims": str(WEAK_DIMS),
                "T":            T,
                "alpha":        ALPHA,
                "epoch":        epoch,
                "train_loss":   round(train_loss, 6),
                "train_acc":    round(train_acc,  6),
                "test_acc":     round(test_acc,   6),
            })
        print()

    temp_csv = "temperature_sweep.csv"
    temp_fields = ["run", "student_dims", "T", "alpha", "epoch", "train_loss", "train_acc", "test_acc"]
    with open(temp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=temp_fields)
        writer.writeheader()
        writer.writerows(temp_rows)
    print(f"=== Saved {temp_csv} ===")

    # =========================================================================
    # Experiment 2 — Model-size sweep  (T fixed at T_BEST)
    # =========================================================================
    size_rows = []

    # Baseline row per student size (trained from scratch at that size)
    for dims in STUDENT_SIZES:
        print(f"=== [Size sweep] Baseline {dims} ===")
        torch.manual_seed(SEED)
        sb = MLP(hidden_dims=dims).to(DEVICE)
        op = torch.optim.Adam(sb.parameters(), lr=LR)
        hist = train_standard(sb, train_loader, op, STUDENT_EPOCHS)
        for (epoch, train_loss, train_acc, test_acc) in hist:
            size_rows.append({
                "run":          "baseline",
                "student_dims": str(dims),
                "T":            "-",
                "alpha":        "-",
                "epoch":        epoch,
                "train_loss":   round(train_loss, 6),
                "train_acc":    round(train_acc,  6),
                "test_acc":     round(test_acc,   6),
            })
        print()

    # Distilled row per student size at T_BEST
    for dims in STUDENT_SIZES:
        print(f"=== [Size sweep] Distilled {dims}  T={T_BEST} ===")
        torch.manual_seed(SEED)
        sk = MLP(hidden_dims=dims).to(DEVICE)
        ok = torch.optim.Adam(sk.parameters(), lr=LR)
        hist = train_distillation(sk, teacher, train_loader, ok,
                                  STUDENT_EPOCHS, T=T_BEST, alpha=ALPHA)
        for (epoch, train_loss, train_acc, test_acc) in hist:
            size_rows.append({
                "run":          "distilled",
                "student_dims": str(dims),
                "T":            T_BEST,
                "alpha":        ALPHA,
                "epoch":        epoch,
                "train_loss":   round(train_loss, 6),
                "train_acc":    round(train_acc,  6),
                "test_acc":     round(test_acc,   6),
            })
        print()

    size_csv = "model_size_sweep.csv"
    size_fields = ["run", "student_dims", "T", "alpha", "epoch", "train_loss", "train_acc", "test_acc"]
    with open(size_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=size_fields)
        writer.writeheader()
        writer.writerows(size_rows)
    print(f"=== Saved {size_csv} ===")

    # --- Combined summary ---
    print(f"\n  Teacher {STRONG_DIMS}: {teacher_acc:.4f}")
    print(f"\n  [Temp sweep]  student={WEAK_DIMS}")
    for T in TEMPERATURES:
        row = next(r for r in temp_rows if r["run"] == "distilled" and r["T"] == T and r["epoch"] == STUDENT_EPOCHS)
        print(f"    T={T:<2}  test_acc={row['test_acc']:.4f}")

    print(f"\n  [Size sweep]  T={T_BEST}")
    for dims in STUDENT_SIZES:
        b = next(r for r in size_rows if r["run"] == "baseline"  and r["student_dims"] == str(dims) and r["epoch"] == STUDENT_EPOCHS)
        k = next(r for r in size_rows if r["run"] == "distilled" and r["student_dims"] == str(dims) and r["epoch"] == STUDENT_EPOCHS)
        print(f"    {str(dims):<12}  baseline={b['test_acc']:.4f}  distilled={k['test_acc']:.4f}  gain={float(k['test_acc'])-float(b['test_acc']):+.4f}")
