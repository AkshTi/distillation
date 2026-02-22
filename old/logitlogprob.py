import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_logits_and_probs(model, loader, device, max_batches=10):
    """Collect logits, probs, logprobs, and true labels for a dataset."""
    model.eval()
    all_logits, all_probs, all_logprobs, all_labels = [], [], [], []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i >= max_batches:
                break
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            logprobs = F.log_softmax(logits, dim=1)
            
            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_logprobs.append(logprobs.cpu())
            all_labels.append(labels)
    
    return {
        'logits':   torch.cat(all_logits),
        'probs':    torch.cat(all_probs),
        'logprobs': torch.cat(all_logprobs),
        'labels':   torch.cat(all_labels),
    }


def plot_confidence_distribution(models_dict, loader, device):
    """
    For each model, plot the distribution of confidence on correct predictions.
    Confidence = probability assigned to the true class.
    This tells you: is the model well-calibrated or overconfident?
    """
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 4))
    if len(models_dict) == 1:
        axes = [axes]
    
    for ax, (name, model) in zip(axes, models_dict.items()):
        data = get_logits_and_probs(model, loader, device)
        # confidence = probability assigned to the TRUE class
        true_class_probs = data['probs'][torch.arange(len(data['labels'])), data['labels']]
        
        ax.hist(true_class_probs.numpy(), bins=50, color='steelblue', edgecolor='white')
        ax.set_xlabel('Confidence on True Class')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}\nmean conf: {true_class_probs.mean():.3f}')
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig('confidence_distributions.png', dpi=150)
    plt.show()


def plot_soft_label_examples(teacher, loader, device, T=3.0, n_examples=5):
    """
    Show what the teacher's soft labels actually look like at different temperatures.
    This makes the abstract concept of 'soft labels' concrete and visual.
    """
    teacher.eval()
    images, labels = next(iter(loader))
    images = images.to(device)
    
    fig, axes = plt.subplots(n_examples, len([1, 3, 10]) + 1, 
                              figsize=(16, 3 * n_examples))
    temperatures = [1.0, 3.0, 10.0]
    
    with torch.no_grad():
        logits = teacher(images)
    
    for i in range(n_examples):
        # Show the image
        axes[i, 0].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[i, 0].set_title(f'True: {labels[i].item()}')
        axes[i, 0].axis('off')
        
        # Show soft label distribution at each temperature
        for j, T in enumerate(temperatures):
            soft = F.softmax(logits[i] / T, dim=0).cpu().numpy()
            axes[i, j+1].bar(range(10), soft, color='salmon')
            axes[i, j+1].set_xticks(range(10))
            axes[i, j+1].set_title(f'T={T}')
            axes[i, j+1].set_xlabel('Class')
            axes[i, j+1].set_ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig('soft_labels_visualization.png', dpi=150)
    plt.show()


def compare_model_agreement(models_dict, loader, device):
    """
    For each pair of models, compute:
    1. Do they agree on the predicted class?
    2. How similar are their full probability distributions (KL divergence)?
    
    This shows whether distillation makes the student 'think like' the teacher
    beyond just getting the same answer.
    """
    data = {}
    for name, model in models_dict.items():
        data[name] = get_logits_and_probs(model, loader, device)
    
    names = list(models_dict.keys())
    print("\n=== Model Agreement Analysis ===")
    
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            
            # Hard label agreement: do they predict the same class?
            pred1 = data[n1]['probs'].argmax(dim=1)
            pred2 = data[n2]['probs'].argmax(dim=1)
            agreement = (pred1 == pred2).float().mean().item()
            
            # Soft distribution agreement: average KL divergence
            # Low KL = distributions are similar = models "think alike"
            kl = F.kl_div(
                data[n1]['logprobs'],
                data[n2]['probs'],
                reduction='batchmean'
            ).item()
            
            print(f"{n1} vs {n2}:")
            print(f"  Prediction agreement: {agreement:.4f}")
            print(f"  Mean KL divergence:   {kl:.4f}  (lower = more similar distributions)")


def plot_error_analysis(models_dict, loader, device):
    """
    Look at what each model gets wrong and whether they make the SAME mistakes.
    If distilled student makes same errors as teacher, that's interesting.
    If it makes different errors, also interesting.
    """
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 4))
    if len(models_dict) == 1:
        axes = [axes]
    
    confusion_matrices = {}
    
    for ax, (name, model) in zip(axes, models_dict.items()):
        data = get_logits_and_probs(model, loader, device, max_batches=50)
        preds = data['probs'].argmax(dim=1)
        labels = data['labels']
        
        # Build confusion matrix
        cm = torch.zeros(10, 10, dtype=torch.long)
        for t, p in zip(labels, preds):
            cm[t][p] += 1
        confusion_matrices[name] = cm
        
        # Plot
        im = ax.imshow(cm.numpy(), cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150)
    plt.show()

    return confusion_matrices


if __name__ == "__main__":
    import os
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEEDS  = [42, 123, 456, 789, 1011]
    MODELS_DIR = "models"

    # Architecture that matches the saved .pt files:
    # network.[0,3,6] are Linear layers; [1,2,4,5] are ReLU+Dropout (no params)
    class StudentMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(784, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128,  64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear( 64,  10),
            )
        def forward(self, x):
            return self.network(x.view(x.size(0), -1))

    def load_student(path):
        m = StudentMLP().to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        return m

    def accuracy(model, loader):
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                correct += (model(imgs).argmax(1) == lbls).sum().item()
                total   += lbls.size(0)
        return correct / total

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=256, shuffle=False,
    )

    # ── Per-seed accuracy summary ─────────────────────────────────────────────
    print("=== Per-seed accuracy ===")
    for seed in SEEDS:
        b = load_student(f"{MODELS_DIR}/student_baseline_seed{seed}.pt")
        d = load_student(f"{MODELS_DIR}/student_distill_seed{seed}.pt")
        b_acc = accuracy(b, test_loader)
        d_acc = accuracy(d, test_loader)
        print(f"  seed {seed:4d}  baseline={b_acc:.4f}  distilled={d_acc:.4f}  Δ={d_acc-b_acc:+.4f}")

    # ── Use seed=42 models for all visual analyses ────────────────────────────
    baseline  = load_student(f"{MODELS_DIR}/student_baseline_seed42.pt")
    distilled = load_student(f"{MODELS_DIR}/student_distill_seed42.pt")

    models = {
        "Baseline (seed42)":  baseline,
        "Distilled (seed42)": distilled,
    }

    print("\n=== Running model agreement analysis ===")
    compare_model_agreement(models, test_loader, DEVICE)

    print("\n=== Plotting confidence distributions ===")
    plot_confidence_distribution(models, test_loader, DEVICE)

    print("\n=== Plotting soft label examples (distilled as pseudo-teacher) ===")
    plot_soft_label_examples(distilled, test_loader, DEVICE, T=3.0, n_examples=5)

    print("\n=== Plotting confusion matrices ===")
    plot_error_analysis(models, test_loader, DEVICE)

    print("\nDone. Saved: confidence_distributions.png, soft_labels_visualization.png, confusion_matrices.png")