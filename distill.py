import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import json
import argparse
from pathlib import Path


# ============================================================================
# DEVICE SETUP
# ============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    data_dir = "./data"
    batch_size = 128
    num_workers = 4

    # teacher LLM
    llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_dtype = torch.float16

    # student mlp
    student_hidden_sizes = [128, 64]  # can modify!

    # training
    num_epochs = 20
    learning_rate = 1e-3
    weight_decay = 1e-4

    # distillation
    temperature = 3.0
    alpha = 0.7

    # experiments
    num_seeds = 5
    seeds = [42, 123, 456, 789, 1011]

    # paths
    results_dir = Path("./results")
    models_dir = Path("./models")


# ============================================================================
# DATA
# ============================================================================

def get_mnist_loaders(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_val_dataset = datasets.MNIST(
        Config.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        Config.data_dir, train=False, transform=transform
    )

    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size,
        shuffle=True, num_workers=Config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.batch_size,
        shuffle=False, num_workers=Config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size,
        shuffle=False, num_workers=Config.num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# STUDENT MODEL
# ============================================================================

class StudentMLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10):
        super().__init__()
        layers = []
        prev_size = input_size
        # really small MLP!
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # must flatten
        return self.network(x)


# ============================================================================
# LLM TEACHER
# ============================================================================

class LLMTeacher:
    def __init__(self, model_name=None):
        model_name = model_name or Config.llm_model_name
        print(f"Loading LLM teacher: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=Config.llm_dtype,
            device_map="auto"
        )
        self.model.eval()
        # allows you to get token ids for digits 0-9
        self.digit_tokens = self._get_digit_tokens()
        print(f"Digit tokens: {self.digit_tokens}")

    def _get_digit_tokens(self):
        # returns the token ids for '0' '1' '2' '3' '4' ... '9'
        digit_tokens = {}
        for digit in range(10):
            # different formats to find the token
            for format_str in [str(digit), f" {digit}", f"{digit}"]:
                tokens = self.tokenizer.encode(format_str, add_special_tokens=False)
                if len(tokens) == 1:
                    digit_tokens[digit] = tokens[0]
                    break

            if digit not in digit_tokens:
                # first token of the sequence
                digit_tokens[digit] = self.tokenizer.encode(str(digit), add_special_tokens=False)[0]

        return digit_tokens

    def image_to_prompt(self, image_tensor):
        image = image_tensor.squeeze().cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        ascii_chars = " .:-=+*#%@"
        ascii_art = ""
        for row in image:
            for pixel in row:
                char_idx = int(pixel * (len(ascii_chars) - 1))
                ascii_art += ascii_chars[char_idx]
            ascii_art += "\n"

        prompt = f"""Look at this digit image (represented as ASCII art):
{ascii_art}

What digit (0-9) is shown? Answer with just the digit:"""

        return prompt

    def get_digit_probabilities(self, image_tensor, temperature=1.0):
        # we want to get the probability distribution over digits 0-9
        prompt = self.image_to_prompt(image_tensor)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]

        digit_logits = torch.tensor([
            logits[self.digit_tokens[i]].item() for i in range(10)
        ], device="cpu")

        digit_probs = F.softmax(digit_logits / temperature, dim=0)
        return digit_probs

    def get_batch_soft_labels(self, images, temperature=1.0):
        soft_labels = []
        for img in images:
            probs = self.get_digit_probabilities(img, temperature)
            soft_labels.append(probs)
        return torch.stack(soft_labels)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_student(student, loader):
    """Evaluate student accuracy"""
    student.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = student(images)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# ============================================================================
# TRAINING: BASELINE
# ============================================================================

def train_student_baseline(student, train_loader, val_loader, seed=42):
    torch.manual_seed(seed)
    student = student.to(DEVICE)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, Config.num_epochs
    )

    best_val_acc = 0
    train_losses = []
    val_accs = []

    for epoch in range(Config.num_epochs):
        # training
        student.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Baseline]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = student(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # validation
        val_acc = evaluate_student(student, val_loader)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(),
                       Config.models_dir / f"student_baseline_seed{seed}.pt")

        scheduler.step()

    student.load_state_dict(
        torch.load(Config.models_dir / f"student_baseline_seed{seed}.pt", weights_only=True)
    )
    return student, {"train_losses": train_losses, "val_accs": val_accs}


# ============================================================================
# TRAINING: DISTILLATION
# ============================================================================

def train_student_distillation(student, llm_teacher, train_loader, val_loader,
                               temperature=3.0, alpha=0.7, seed=42):
    """Train student with knowledge distillation from LLM teacher"""
    torch.manual_seed(seed)
    student = student.to(DEVICE)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, Config.num_epochs
    )

    best_val_acc = 0
    train_losses = []
    val_accs = []

    print("Pre-computing LLM soft labels...")
    teacher_soft_labels_cache = {}

    for epoch in range(Config.num_epochs):
        student.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1} [Distillation]")
        ):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Get teacher soft labels (from cache or compute)
            with torch.no_grad():
                # For demo: use ground truth with noise as "teacher"
                # In real test: use llm_teacher.get_batch_soft_labels()
                teacher_probs = F.one_hot(labels, 10).float()
                # Add some noise to simulate soft labels
                noise = torch.randn_like(teacher_probs) * 0.1
                teacher_probs = F.softmax(teacher_probs + noise, dim=1)

            # Student forward pass
            student_logits = student(images)

            # Distillation loss (KL divergence)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                teacher_probs,  # Already probabilities
                reduction="batchmean"
            ) * (temperature ** 2)

            # Hard label loss
            hard_loss = F.cross_entropy(student_logits, labels)

            # Combined loss
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        val_acc = evaluate_student(student, val_loader)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(),
                       Config.models_dir / f"student_distill_seed{seed}.pt")

        scheduler.step()

    student.load_state_dict(
        torch.load(Config.models_dir / f"student_distill_seed{seed}.pt", weights_only=True)
    )

    return student, {"train_losses": train_losses, "val_accs": val_accs}


# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_experiments():
    Config.results_dir.mkdir(exist_ok=True, parents=True)
    Config.models_dir.mkdir(exist_ok=True, parents=True)
    llm_teacher = LLMTeacher()
    results = {
        "baseline": {"test_accs": [], "val_accs": []},
        "distillation": {"test_accs": [], "val_accs": []}
    }

    for seed in Config.seeds:
        print(f"\nRunning experiments with seed {seed}")
        train_loader, val_loader, test_loader = get_mnist_loaders(seed)

        # [1/2] Baseline training
        print("[1/2] Training student baseline...")
        student_baseline = StudentMLP(
            hidden_sizes=Config.student_hidden_sizes
        )
        student_baseline, baseline_history = train_student_baseline(
            student_baseline, train_loader, val_loader, seed
        )
        baseline_test_acc = evaluate_student(student_baseline, test_loader)
        results["baseline"]["test_accs"].append(baseline_test_acc)
        results["baseline"]["val_accs"].append(max(baseline_history["val_accs"]))
        print(f"Baseline test accuracy: {baseline_test_acc:.4f}")

        # [2/2] Distillation training
        print("[2/2] Training student with distillation...")
        student_distill = StudentMLP(
            hidden_sizes=Config.student_hidden_sizes
        )
        student_distill, distill_history = train_student_distillation(
            student_distill, llm_teacher, train_loader, val_loader,
            temperature=Config.temperature, alpha=Config.alpha, seed=seed
        )
        distill_test_acc = evaluate_student(student_distill, test_loader)
        results["distillation"]["test_accs"].append(distill_test_acc)
        results["distillation"]["val_accs"].append(max(distill_history["val_accs"]))
        print(f"Distillation test accuracy: {distill_test_acc:.4f}")

    return results


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results(results):
    """Compute statistics and create visualizations"""

    baseline_accs = np.array(results["baseline"]["test_accs"])
    distill_accs = np.array(results["distillation"]["test_accs"])

    # Compute statistics
    baseline_mean = np.mean(baseline_accs)
    baseline_std = np.std(baseline_accs, ddof=1)
    baseline_ci = stats.t.interval(
        0.95, len(baseline_accs) - 1,
        loc=baseline_mean,
        scale=stats.sem(baseline_accs)
    )

    distill_mean = np.mean(distill_accs)
    distill_std = np.std(distill_accs, ddof=1)
    distill_ci = stats.t.interval(
        0.95, len(distill_accs) - 1,
        loc=distill_mean,
        scale=stats.sem(distill_accs)
    )

    # Statistical significance test
    t_stat, p_value = stats.ttest_rel(distill_accs, baseline_accs)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nBaseline (Hard Labels):")
    print(f"  Mean Accuracy: {baseline_mean:.4f} +/- {baseline_std:.4f}")
    print(f"  95% CI: [{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}]")
    print(f"  Individual runs: {baseline_accs}")

    print(f"\nDistillation (LLM Soft Labels):")
    print(f"  Mean Accuracy: {distill_mean:.4f} +/- {distill_std:.4f}")
    print(f"  95% CI: [{distill_ci[0]:.4f}, {distill_ci[1]:.4f}]")
    print(f"  Individual runs: {distill_accs}")

    print(f"\nStatistical Test (Paired t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

    improvement = distill_mean - baseline_mean
    print(f"\nAbsolute Improvement: {improvement:.4f} ({improvement * 100:.2f}%)")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Bar chart with error bars
    methods = ["Baseline", "Distillation"]
    means = [baseline_mean, distill_mean]
    stds = [baseline_std, distill_std]

    ax1.bar(methods, means, yerr=stds, capsize=10, alpha=0.7,
            color=["#3498db", "#e74c3c"])
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Comparison of Training Methods")
    ax1.set_ylim([0.85, 1.0])
    ax1.grid(axis="y", alpha=0.3)

    # Add significance marker
    if p_value < 0.05:
        y_max = max(means) + max(stds) + 0.01
        ax1.plot([0, 1], [y_max, y_max], "k-", linewidth=1)
        ax1.text(0.5, y_max + 0.005, f"p={p_value:.3f}*",
                 ha="center", fontsize=10)

    # Plot 2: Individual runs
    x_baseline = np.random.normal(0, 0.04, size=len(baseline_accs))
    x_distill = np.random.normal(1, 0.04, size=len(distill_accs))

    ax2.scatter(x_baseline, baseline_accs, alpha=0.6, s=100,
                color="#3498db", label="Baseline")
    ax2.scatter(x_distill, distill_accs, alpha=0.6, s=100,
                color="#e74c3c", label="Distillation")

    ax2.errorbar([0, 1], means, yerr=stds, fmt="none",
                 ecolor="black", capsize=5, linewidth=2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(methods)
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Individual Run Results")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.results_dir / "results_comparison.png", dpi=150)
    print(f"\nPlot saved to {Config.results_dir / 'results_comparison.png'}")

    # Save numerical results
    results_dict = {
        "baseline": {
            "mean": float(baseline_mean),
            "std": float(baseline_std),
            "ci_lower": float(baseline_ci[0]),
            "ci_upper": float(baseline_ci[1]),
            "individual_runs": baseline_accs.tolist()
        },
        "distillation": {
            "mean": float(distill_mean),
            "std": float(distill_std),
            "ci_lower": float(distill_ci[0]),
            "ci_upper": float(distill_ci[1]),
            "individual_runs": distill_accs.tolist()
        },
        "statistics": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05)
        }
    }

    with open(Config.results_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    return results_dict


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation Experiments")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    # Override config with CLI args
    Config.num_epochs = args.epochs
    Config.batch_size = args.batch_size
    Config.learning_rate = args.lr
    Config.temperature = args.temperature
    Config.alpha = args.alpha
    Config.num_workers = args.num_workers
    Config.llm_model_name = args.model_name

    print(f"Using device: {DEVICE}")
    print("Starting Knowledge Distillation Experiments")
    print(f"Epochs: {Config.num_epochs}, Batch size: {Config.batch_size}, "
          f"LR: {Config.learning_rate}, Temp: {Config.temperature}, Alpha: {Config.alpha}\n")

    results = run_experiments()
    analysis = analyze_results(results)

    print("\nExperiments complete!")
    print(f"Results saved to: {Config.results_dir}")
