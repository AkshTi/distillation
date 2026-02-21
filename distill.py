import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3
TEMPERATURE = 3.0
ALPHA = 0.5  # weight on soft loss; (1-alpha) goes to hard loss
SEED = 42

torch.manual_seed(SEED)

# ── Data ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model ────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        """
        hidden_sizes: list of ints, e.g. [512, 512] for teacher, [64] for student
        """
        super().__init__()
        layers = []
        in_size = 784  # 28*28 flattened
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 10))  # 10 classes
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)  # returns raw logits


# ── Loss functions ────────────────────────────────────────────────────────────
def cross_entropy_loss(student_logits, true_labels):
    return F.cross_entropy(student_logits, true_labels)


def distillation_loss(student_logits, teacher_logits, true_labels, T=TEMPERATURE, alpha=ALPHA):
    # Soft loss: match the teacher's soft distribution
    # We divide by T to soften, then T^2 to rescale gradients back to normal magnitude
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),   # student log-probs
        F.softmax(teacher_logits / T, dim=1),        # teacher probs (target)
        reduction='batchmean'
    ) * (T ** 2)

    # Hard loss: still learn from true labels
    hard_loss = F.cross_entropy(student_logits, true_labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


# ── Training ──────────────────────────────────────────────────────────────────
def train_standard(model, epochs=EPOCHS):
    """Train with cross-entropy on true labels only."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    history = []

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = cross_entropy_loss(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model)
        history.append({'epoch': epoch+1, 'loss': avg_loss, 'acc': acc})
        print(f"[Standard] Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    return history


def train_distillation(student, teacher, epochs=EPOCHS, T=TEMPERATURE, alpha=ALPHA):
    """Train student to match teacher soft outputs + true labels."""
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    teacher.eval()  # teacher is frozen
    student.train()
    history = []

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(images)  # don't update teacher

            optimizer.zero_grad()
            student_logits = student(images)
            loss = distillation_loss(student_logits, teacher_logits, labels, T=T, alpha=alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(student)
        history.append({'epoch': epoch+1, 'loss': avg_loss, 'acc': acc})
        print(f"[Distill]  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    return history


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total


# ── Main experiment ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Train teacher (big model)
    print("=== Training Teacher ===")
    teacher = MLP(hidden_sizes=[512, 512]).to(DEVICE)
    train_standard(teacher, epochs=EPOCHS)
    teacher_acc = evaluate(teacher)
    print(f"Teacher accuracy: {teacher_acc:.4f}")
    torch.save(teacher.state_dict(), "teacher.pt")

    # 2. Train baseline student (small model, standard loss)
    print("\n=== Training Baseline Student (no distillation) ===")
    baseline_student = MLP(hidden_sizes=[64]).to(DEVICE)
    train_standard(baseline_student, epochs=EPOCHS)
    baseline_acc = evaluate(baseline_student)
    print(f"Baseline student accuracy: {baseline_acc:.4f}")

    # 3. Train distilled student (same small model, distillation loss)
    print("\n=== Training Distilled Student ===")
    distilled_student = MLP(hidden_sizes=[64]).to(DEVICE)
    distill_history = train_distillation(distilled_student, teacher, epochs=EPOCHS)
    distilled_acc = evaluate(distilled_student)
    print(f"Distilled student accuracy: {distilled_acc:.4f}")

    # 4. Results summary
    print("\n=== Results ===")
    print(f"Teacher accuracy:          {teacher_acc:.4f}")
    print(f"Baseline student accuracy: {baseline_acc:.4f}")
    print(f"Distilled student accuracy:{distilled_acc:.4f}")
    print(f"Distillation gap closed:   {(distilled_acc - baseline_acc) / (teacher_acc - baseline_acc) * 100:.1f}%")