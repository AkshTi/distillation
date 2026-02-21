import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
LR = 1e-3
SEED = 42

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
# Pass hidden_dims to switch between strong/weak models, e.g.:
#   STRONG_DIMS = [512, 512]
#   WEAK_DIMS   = [128, 128]
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
# Training helpers
# ---------------------------------------------------------------------------
def train_standard(model, loader, optimizer, epochs: int):
    model.train()
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
        print(f"  [Standard] Epoch {epoch:>2} | loss {total_loss/len(loader.dataset):.4f} | acc {correct/len(loader.dataset):.4f}")


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


# ---------------------------------------------------------------------------
# Distillation loss  (Hinton et al. 2015)
#
# L = alpha * CE(student_logits, hard_labels)
#   + (1 - alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T))
#
# The T^2 factor re-scales the gradient to match the magnitude it would have
# at temperature 1, as described in the original paper.
# ---------------------------------------------------------------------------
def distillation_loss(student_logits, teacher_logits, labels, T: float, alpha: float):
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)

    kd_loss  = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    ce_loss  = F.cross_entropy(student_logits, labels)
    return alpha * ce_loss + (1.0 - alpha) * kd_loss


def train_distillation(student, teacher, loader, optimizer,
                        epochs: int, T: float = 4.0, alpha: float = 0.1):
    teacher.eval()
    student.train()
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
        print(f"  [Distill]  Epoch {epoch:>2} | loss {total_loss/len(loader.dataset):.4f} | acc {correct/len(loader.dataset):.4f}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)

    TEACHER_EPOCHS  = 10
    STUDENT_EPOCHS  = 10
    TEMPERATURE     = 4.0   # softens teacher distribution
    ALPHA           = 0.1   # weight on hard-label CE loss

    STRONG_DIMS = [512, 512]   # teacher (stronger model)
    WEAK_DIMS   = [128, 128]   # student (weaker model)

    # --- Train teacher ---
    print("=== Training teacher (strong model) ===")
    teacher = MLP(hidden_dims=STRONG_DIMS).to(DEVICE)
    opt_t   = torch.optim.Adam(teacher.parameters(), lr=LR)
    train_standard(teacher, train_loader, opt_t, TEACHER_EPOCHS)
    teacher_acc = evaluate(teacher, test_loader)
    print(f"  Teacher test acc: {teacher_acc:.4f}\n")

    # --- Train student from scratch (baseline) ---
    print("=== Training student from scratch (baseline) ===")
    student_baseline = MLP(hidden_dims=WEAK_DIMS).to(DEVICE)
    opt_sb = torch.optim.Adam(student_baseline.parameters(), lr=LR)
    train_standard(student_baseline, train_loader, opt_sb, STUDENT_EPOCHS)
    baseline_acc = evaluate(student_baseline, test_loader)
    print(f"  Student (baseline) test acc: {baseline_acc:.4f}\n")

    # --- Train student via distillation ---
    print(f"=== Distilling teacher → student  (T={TEMPERATURE}, alpha={ALPHA}) ===")
    student_kd = MLP(hidden_dims=WEAK_DIMS).to(DEVICE)
    opt_kd     = torch.optim.Adam(student_kd.parameters(), lr=LR)
    train_distillation(student_kd, teacher, train_loader, opt_kd,
                       STUDENT_EPOCHS, T=TEMPERATURE, alpha=ALPHA)
    kd_acc = evaluate(student_kd, test_loader)
    print(f"  Student (distilled) test acc: {kd_acc:.4f}\n")

    # --- Summary ---
    print("=== Results ===")
    print(f"  Teacher  {STRONG_DIMS}: {teacher_acc:.4f}")
    print(f"  Student  {WEAK_DIMS} (scratch):      {baseline_acc:.4f}")
    print(f"  Student  {WEAK_DIMS} (distilled):    {kd_acc:.4f}")
    print(f"  Distillation gain: {kd_acc - baseline_acc:+.4f}")
