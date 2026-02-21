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
    WEAK_DIMS   = [128, 128]   # student

    TEMPERATURES = list(range(1, 11))   # T = 1, 2, ..., 10

    # --- Train teacher once ---
    print("=== Training teacher (strong model) ===")
    teacher = MLP(hidden_dims=STRONG_DIMS).to(DEVICE)
    opt_t   = torch.optim.Adam(teacher.parameters(), lr=LR)
    train_standard(teacher, train_loader, opt_t, TEACHER_EPOCHS)
    teacher_acc = evaluate(teacher, test_loader)
    print(f"  Teacher test acc: {teacher_acc:.4f}\n")

    # --- Train student baseline once ---
    print("=== Training student from scratch (baseline) ===")
    torch.manual_seed(SEED)
    student_baseline = MLP(hidden_dims=WEAK_DIMS).to(DEVICE)
    opt_sb = torch.optim.Adam(student_baseline.parameters(), lr=LR)
    baseline_history = train_standard(student_baseline, train_loader, opt_sb, STUDENT_EPOCHS)
    baseline_acc = evaluate(student_baseline, test_loader)
    print(f"  Student (baseline) test acc: {baseline_acc:.4f}\n")

    # --- Sweep temperatures ---
    csv_rows = []

    # Baseline rows (T=0 sentinel so it sorts cleanly)
    for (epoch, train_loss, train_acc, test_acc) in baseline_history:
        csv_rows.append({
            "run":        "baseline",
            "T":          0,
            "alpha":      "-",
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "test_acc":   round(test_acc,   6),
        })

    for T in TEMPERATURES:
        print(f"=== Distilling  T={T}  alpha={ALPHA} ===")
        torch.manual_seed(SEED)
        student_kd = MLP(hidden_dims=WEAK_DIMS).to(DEVICE)
        opt_kd     = torch.optim.Adam(student_kd.parameters(), lr=LR)
        history    = train_distillation(student_kd, teacher, train_loader, opt_kd,
                                        STUDENT_EPOCHS, T=T, alpha=ALPHA)
        for (epoch, train_loss, train_acc, test_acc) in history:
            csv_rows.append({
                "run":        "distilled",
                "T":          T,
                "alpha":      ALPHA,
                "epoch":      epoch,
                "train_loss": round(train_loss, 6),
                "train_acc":  round(train_acc,  6),
                "test_acc":   round(test_acc,   6),
            })
        print()

    # --- Write CSV ---
    csv_path = "temperature_sweep.csv"
    fieldnames = ["run", "T", "alpha", "epoch", "train_loss", "train_acc", "test_acc"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"=== Saved results to {csv_path} ===")
    print(f"  Teacher          {STRONG_DIMS}: {teacher_acc:.4f}")
    print(f"  Student baseline {WEAK_DIMS}:   {baseline_acc:.4f}")
    for T in TEMPERATURES:
        final = [r for r in csv_rows if r["run"] == "distilled" and r["T"] == T and r["epoch"] == STUDENT_EPOCHS]
        if final:
            print(f"  Student distilled T={T:<2}:          {final[0]['test_acc']:.4f}")
