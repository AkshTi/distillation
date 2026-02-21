import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 10
SEED = 42
torch.manual_seed(SEED)

# ── Data ─────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

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

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# ── Training utilities ────────────────────────────────────────────────────────
def train_standard(model, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            F.cross_entropy(model(images), labels).backward()
            optimizer.step()
    return evaluate(model)


def train_on_weak_labels(strong_model, weak_model, epochs=EPOCHS, use_soft_labels=True, T=3.0):
    """
    Train a strong model supervised only by a weak model's outputs.
    
    use_soft_labels=True:  student matches weak model's full soft distribution (distillation style)
    use_soft_labels=False: student matches weak model's hard predicted labels (simulates noisy labeler)
    """
    optimizer = torch.optim.Adam(strong_model.parameters(), lr=LR)
    weak_model.eval()

    for epoch in range(epochs):
        strong_model.train()
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.no_grad():
                weak_logits = weak_model(images)
                weak_hard_labels = weak_logits.argmax(dim=1)  # weak model's predicted class

            optimizer.zero_grad()
            strong_logits = strong_model(images)

            if use_soft_labels:
                # Match weak model's full soft distribution — more signal
                loss = F.kl_div(
                    F.log_softmax(strong_logits / T, dim=1),
                    F.softmax(weak_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T ** 2)
            else:
                # Match only weak model's argmax prediction — simulates a human labeler who is often wrong
                loss = F.cross_entropy(strong_logits, weak_hard_labels)

            loss.backward()
            optimizer.step()

    return evaluate(strong_model)


def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ── Experiment ────────────────────────────────────────────────────────────────
# We define a chain of models: tiny -> small -> medium -> large
# Each is trained on supervision from the previous (weaker) one
# Key question: does a strong model recover capability beyond its weak supervisor?

model_configs = {
    'tiny':   [16],
    'small':  [64],
    'medium': [256],
    'large':  [512, 512],
}

N_SEEDS = 3
results = []

for seed in range(N_SEEDS):
    torch.manual_seed(seed)
    print(f"\n=== Seed {seed} ===")

    # Step 1: train each model on TRUE labels — this is your ceiling
    print("Training all models on true labels (ceiling)...")
    ceiling = {}
    trained_models = {}
    for name, hidden in model_configs.items():
        m = MLP(hidden).to(DEVICE)
        acc = train_standard(m, epochs=EPOCHS)
        ceiling[name] = acc
        trained_models[name] = m
        print(f"  {name} ceiling: {acc:.4f}  ({count_params(m):,} params)")

    # Step 2: train each model supervised ONLY by the previous (weaker) model
    # tiny trains on true labels (it IS the weak supervisor at the start)
    # small trains on tiny's outputs
    # medium trains on small's outputs
    # large trains on medium's outputs
    print("\nTraining chain: each model supervised by weaker model...")
    
    names = list(model_configs.keys())
    weak_supervised_accs = {}
    
    # Tiny trains normally — it's the root weak supervisor
    tiny_model = MLP(model_configs['tiny']).to(DEVICE)
    tiny_acc = train_standard(tiny_model, epochs=EPOCHS)
    weak_supervised_accs['tiny'] = tiny_acc
    current_weak = tiny_model
    print(f"  tiny (root supervisor): {tiny_acc:.4f}")

    for name in names[1:]:  # small, medium, large
        strong_model = MLP(model_configs[name]).to(DEVICE)
        acc = train_on_weak_labels(strong_model, current_weak, use_soft_labels=True)
        weak_supervised_accs[name] = acc
        print(f"  {name} (supervised by previous): {acc:.4f}  ceiling was {ceiling[name]:.4f}")
        current_weak = strong_model  # this model becomes the next supervisor

    # Step 3: compute how much of the ceiling each model recovered
    for name in names:
        gap_closed = None
        if name != 'tiny':
            gap = ceiling[name] - ceiling['tiny']  # how much better could it be vs root supervisor
            recovered = weak_supervised_accs[name] - ceiling['tiny']
            gap_closed = recovered / gap if gap > 0 else None

        results.append({
            'seed': seed,
            'model': name,
            'params': count_params(MLP(model_configs[name])),
            'ceiling_acc': ceiling[name],
            'weak_supervised_acc': weak_supervised_accs[name],
            'gap_closed_pct': gap_closed * 100 if gap_closed is not None else None,
        })


# ── Plot results ──────────────────────────────────────────────────────────────
import pandas as pd
df = pd.DataFrame(results)

names = list(model_configs.keys())
x = np.arange(len(names))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: ceiling vs weak-supervised accuracy per model
ax = axes[0]
ceiling_means = [df[df.model == n]['ceiling_acc'].mean() for n in names]
ceiling_stds  = [df[df.model == n]['ceiling_acc'].std() for n in names]
weak_means    = [df[df.model == n]['weak_supervised_acc'].mean() for n in names]
weak_stds     = [df[df.model == n]['weak_supervised_acc'].std() for n in names]

ax.bar(x - width/2, ceiling_means, width, yerr=ceiling_stds, label='Ceiling (true labels)', color='steelblue', capsize=4)
ax.bar(x + width/2, weak_means,    width, yerr=weak_stds,    label='Weak supervised',       color='salmon',   capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel('Test Accuracy')
ax.set_title('Ceiling vs Weak-Supervised Accuracy')
ax.legend()
# Don't start y-axis at 0 — differences are small
min_val = min(min(ceiling_means), min(weak_means))
ax.set_ylim(min_val - 0.02, 1.0)

# Plot 2: % of ceiling gap recovered, by model size
ax = axes[1]
non_tiny = [n for n in names if n != 'tiny']
gap_means = [df[df.model == n]['gap_closed_pct'].mean() for n in non_tiny]
gap_stds  = [df[df.model == n]['gap_closed_pct'].std()  for n in non_tiny]
params    = [df[df.model == n]['params'].iloc[0] for n in non_tiny]

ax.errorbar(params, gap_means, yerr=gap_stds, fmt='o-', capsize=4, color='purple')
ax.axhline(100, linestyle='--', color='gray', label='Full recovery (100%)')
ax.set_xscale('log')
ax.set_xlabel('Model Parameters (log scale)')
ax.set_ylabel('% of Ceiling Gap Recovered')
ax.set_title('Does Larger Model Recover More from Weak Supervision?')
ax.legend()

plt.tight_layout()
plt.savefig('weak_to_strong_mnist.png', dpi=150)
plt.show()

print("\n=== Summary Table ===")
summary = df.groupby('model').agg({
    'ceiling_acc': ['mean', 'std'],
    'weak_supervised_acc': ['mean', 'std'],
    'gap_closed_pct': ['mean', 'std']
}).round(4)
print(summary)