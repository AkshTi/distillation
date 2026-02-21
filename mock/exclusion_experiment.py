import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
LR         = 1e-3
SEED       = 42

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

test_loader_full = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# Configurable MLP
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
# Dataset filtering
# ---------------------------------------------------------------------------
def exclude_digit(dataset, excluded: int):
    targets = torch.tensor(dataset.targets)
    indices = (targets != excluded).nonzero(as_tuple=True)[0].tolist()
    return Subset(dataset, indices)


def only_digit(dataset, digit: int):
    targets = torch.tensor(dataset.targets)
    indices = (targets == digit).nonzero(as_tuple=True)[0].tolist()
    return Subset(dataset, indices)


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


def train_standard(model, train_loader, optimizer, epochs: int,
                   test_loader_seen=None, test_loader_excl=None, label="Standard"):
    """
    Train with hard labels.
    Returns list of (epoch, train_loss, train_acc, test_acc_seen, test_acc_excl).
    test_loader_seen : full test set (9 seen digits)
    test_loader_excl : test set containing ONLY the excluded digit
    """
    history = []
    for epoch in range(1, epochs + 1):
        total_loss, correct = 0.0, 0
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        train_loss    = total_loss / len(train_loader.dataset)
        train_acc     = correct   / len(train_loader.dataset)
        acc_seen      = evaluate(model, test_loader_seen) if test_loader_seen else float("nan")
        acc_excl      = evaluate(model, test_loader_excl) if test_loader_excl else float("nan")
        history.append((epoch, train_loss, train_acc, acc_seen, acc_excl))
        print(f"  [{label}] Epoch {epoch:>2} | loss {train_loss:.4f} | "
              f"train {train_acc:.4f} | seen {acc_seen:.4f} | excl {acc_excl:.4f}")
    return history


# ---------------------------------------------------------------------------
# Distillation loss  (Hinton et al. 2015)
# ---------------------------------------------------------------------------
def distillation_loss(student_logits, teacher_logits, labels, T: float, alpha: float):
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * ce_loss + (1.0 - alpha) * kd_loss


def train_distillation(student, teacher, train_loader, optimizer,
                       epochs: int, T: float, alpha: float,
                       test_loader_seen=None, test_loader_excl=None, label="Distill"):
    """
    Train student via distillation from teacher.
    Returns list of (epoch, train_loss, train_acc, test_acc_seen, test_acc_excl).
    """
    teacher.eval()
    history = []
    for epoch in range(1, epochs + 1):
        total_loss, correct = 0.0, 0
        student.train()
        for x, y in train_loader:
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
        train_loss = total_loss / len(train_loader.dataset)
        train_acc  = correct   / len(train_loader.dataset)
        acc_seen   = evaluate(student, test_loader_seen) if test_loader_seen else float("nan")
        acc_excl   = evaluate(student, test_loader_excl) if test_loader_excl else float("nan")
        history.append((epoch, train_loss, train_acc, acc_seen, acc_excl))
        print(f"  [{label}] Epoch {epoch:>2} | loss {train_loss:.4f} | "
              f"train {train_acc:.4f} | seen {acc_seen:.4f} | excl {acc_excl:.4f}")
    return history


def record(rows, history, excluded, model_name, T, alpha):
    for (epoch, train_loss, train_acc, acc_seen, acc_excl) in history:
        rows.append({
            "excluded_digit":    excluded,
            "model":             model_name,
            "T":                 T,
            "alpha":             alpha,
            "epoch":             epoch,
            "train_loss":        round(train_loss, 6),
            "train_acc":         round(train_acc,  6),
            "test_acc_seen":     round(acc_seen,   6),
            "test_acc_excluded": round(acc_excl,   6),
        })


# ---------------------------------------------------------------------------
# Main experiment
#
# For each excluded digit d in 0..9:
#   1. Teacher  [512,512] trained on MNIST \ {d}
#   2. Baseline [128,128] trained on MNIST \ {d}  (standard CE)
#   3. Student  [128,128] trained on MNIST \ {d}  (distilled from teacher, T=3)
#
#   All three are evaluated on:
#     - test_acc_seen : full test set (proxy for performance on known digits)
#     - test_acc_excluded : test samples of digit d only
#       (measures how the model handles a completely unseen digit class)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)

    TEACHER_EPOCHS = 10
    STUDENT_EPOCHS = 10
    ALPHA          = 0.1
    T_FIXED        = 8

    STRONG_DIMS = [512, 512]
    WEAK_DIMS   = [128, 128]

    ALL_DIGITS = list(range(10))
    csv_rows   = []

    for excluded in ALL_DIGITS:
        print(f"\n{'='*65}")
        print(f"  Excluded digit: {excluded}")
        print(f"{'='*65}")

        # Filtered train (no digit d) + test of only digit d
        train_filtered       = exclude_digit(train_dataset, excluded)
        test_excl_only       = only_digit(test_dataset, excluded)
        train_loader_filt    = DataLoader(train_filtered, batch_size=BATCH_SIZE, shuffle=True)
        test_loader_excl     = DataLoader(test_excl_only, batch_size=BATCH_SIZE, shuffle=False)

        # ---- Teacher (strong model, same filtered set) ----
        print(f"\n  -- Teacher [512,512]  (digit {excluded} excluded) --")
        torch.manual_seed(SEED)
        teacher = MLP(hidden_dims=STRONG_DIMS).to(DEVICE)
        opt_t   = torch.optim.Adam(teacher.parameters(), lr=LR)
        t_hist  = train_standard(
            teacher, train_loader_filt, opt_t, TEACHER_EPOCHS,
            test_loader_seen=test_loader_full,
            test_loader_excl=test_loader_excl,
            label=f"Teacher excl={excluded}",
        )
        record(csv_rows, t_hist, excluded, "teacher", "-", "-")

        # ---- Baseline student (standard CE, same filtered set) ----
        print(f"\n  -- Baseline [128,128]  (digit {excluded} excluded) --")
        torch.manual_seed(SEED)
        student_base = MLP(hidden_dims=WEAK_DIMS).to(DEVICE)
        opt_base     = torch.optim.Adam(student_base.parameters(), lr=LR)
        base_hist    = train_standard(
            student_base, train_loader_filt, opt_base, STUDENT_EPOCHS,
            test_loader_seen=test_loader_full,
            test_loader_excl=test_loader_excl,
            label=f"Base excl={excluded}",
        )
        record(csv_rows, base_hist, excluded, "baseline", "-", "-")

        # ---- Distilled student (T=3, from teacher above) ----
        print(f"\n  -- Distilled [128,128] T={T_FIXED}  (digit {excluded} excluded) --")
        torch.manual_seed(SEED)
        student_kd = MLP(hidden_dims=WEAK_DIMS).to(DEVICE)
        opt_kd     = torch.optim.Adam(student_kd.parameters(), lr=LR)
        kd_hist    = train_distillation(
            student_kd, teacher, train_loader_filt, opt_kd,
            STUDENT_EPOCHS, T=T_FIXED, alpha=ALPHA,
            test_loader_seen=test_loader_full,
            test_loader_excl=test_loader_excl,
            label=f"KD T={T_FIXED} excl={excluded}",
        )
        record(csv_rows, kd_hist, excluded, "distilled", T_FIXED, ALPHA)

    # --- Write CSV ---
    csv_path   = "exclusion_experiment.csv"
    fieldnames = [
        "excluded_digit", "model", "T", "alpha",
        "epoch", "train_loss", "train_acc",
        "test_acc_seen", "test_acc_excluded",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n=== Saved results to {csv_path} ===")

    # Summary: final-epoch performance on the excluded digit for each model/digit
    print(f"\n{'digit':<8} {'teacher (excl)':<18} {'baseline (excl)':<18} {'distilled T=3 (excl)':<22} {'KD gain'}")
    print("-" * 75)
    for excluded in ALL_DIGITS:
        def final(model_name):
            return next(
                r for r in csv_rows
                if r["excluded_digit"] == excluded
                and r["model"] == model_name
                and r["epoch"] == STUDENT_EPOCHS
            )
        t = float(final("teacher")  ["test_acc_excluded"])
        b = float(final("baseline") ["test_acc_excluded"])
        k = float(final("distilled")["test_acc_excluded"])
        print(f"  {excluded:<6}   {t:<18.4f} {b:<18.4f} {k:<22.4f} {k-b:+.4f}")
