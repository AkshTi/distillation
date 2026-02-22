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
N_SEEDS = 3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

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

def train_and_eval(model, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        val_accs.append(evaluate(model))

    return train_losses, val_accs

def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ── Model configs to sweep ────────────────────────────────────────────────────
# Two axes of scaling: width and depth
# This lets you ask: does scaling width vs depth behave differently?

configs = {
    # width scaling (fixed 1 layer, increasing width)
    'w16':       [16],
    'w32':       [32],
    'w64':       [64],
    'w128':      [128],
    'w256':      [256],
    'w512':      [512],
    # depth scaling (fixed width=128, increasing depth)
    'd1_w128':   [128],
    'd2_w128':   [128, 128],
    'd3_w128':   [128, 128, 128],
    'd4_w128':   [128, 128, 128, 128],
}

results = []

for seed in range(N_SEEDS):
    torch.manual_seed(seed)
    print(f"\n=== Seed {seed} ===")

    for name, hidden in configs.items():
        model = MLP(hidden).to(DEVICE)
        n_params = count_params(model)
        train_losses, val_accs = train_and_eval(model, epochs=EPOCHS)

        final_acc = val_accs[-1]
        final_loss = train_losses[-1]

        results.append({
            'name':        name,
            'hidden':      hidden,
            'n_params':    n_params,
            'seed':        seed,
            'final_acc':   final_acc,
            'final_loss':  final_loss,
            'val_accs':    val_accs,
            'train_losses': train_losses,
            'scaling_type': 'width' if name.startswith('w') else 'depth',
        })

        print(f"  {name:12s} | params={n_params:7,} | acc={final_acc:.4f}")

# ── Plotting ──────────────────────────────────────────────────────────────────
import pandas as pd
df = pd.DataFrame(results)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── Plot 1: accuracy vs param count (width scaling) ───────────────────────────
ax = axes[0, 0]
width_df = df[df.scaling_type == 'width']
params   = sorted(width_df['n_params'].unique())
means    = [width_df[width_df.n_params == p]['final_acc'].mean() for p in params]
stds     = [width_df[width_df.n_params == p]['final_acc'].std()  for p in params]

ax.errorbar(params, means, yerr=stds, fmt='o-', capsize=4, color='steelblue')
ax.set_xscale('log')
ax.set_xlabel('Parameter Count (log scale)')
ax.set_ylabel('Test Accuracy')
ax.set_title('Width Scaling: Accuracy vs Parameters')
ax.set_ylim(min(means) - 0.01, 1.0)

# ── Plot 2: accuracy vs depth ─────────────────────────────────────────────────
ax = axes[0, 1]
depth_df = df[df.scaling_type == 'depth']
depths   = [1, 2, 3, 4]
depth_names = ['d1_w128', 'd2_w128', 'd3_w128', 'd4_w128']
means    = [depth_df[depth_df.name == n]['final_acc'].mean() for n in depth_names]
stds     = [depth_df[depth_df.name == n]['final_acc'].std()  for n in depth_names]

ax.errorbar(depths, means, yerr=stds, fmt='s-', capsize=4, color='salmon')
ax.set_xlabel('Number of Hidden Layers')
ax.set_ylabel('Test Accuracy')
ax.set_title('Depth Scaling: Accuracy vs Layers (width=128 fixed)')
ax.set_xticks(depths)
ax.set_ylim(min(means) - 0.01, 1.0)

# ── Plot 3: learning curves for width scaling ──────────────────────────────────
ax = axes[1, 0]
colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(params)))
for color, p in zip(colors, params):
    subset = width_df[width_df.n_params == p]
    # average val_accs across seeds
    avg_curve = np.mean([r for r in subset['val_accs']], axis=0)
    label = f"{p:,} params"
    ax.plot(range(1, EPOCHS+1), avg_curve, color=color, label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Learning Curves: Width Scaling')
ax.legend(fontsize=7)

# ── Plot 4: width vs depth comparison at similar param counts ──────────────────
ax = axes[1, 1]
# d2_w128 has similar params to w256 — compare them directly
compare = {
    'w256 (wide, shallow)': 'w256',
    'd2_w128 (narrow, deep)': 'd2_w128',
    'w512 (wider)': 'w512',
    'd3_w128 (deeper)': 'd3_w128',
}
x = np.arange(len(compare))
accs  = [df[df.name == v]['final_acc'].mean() for v in compare.values()]
stds  = [df[df.name == v]['final_acc'].std()  for v in compare.values()]
params_list = [df[df.name == v]['n_params'].iloc[0] for v in compare.values()]

bars = ax.bar(x, accs, yerr=stds, capsize=4,
              color=['steelblue', 'salmon', 'steelblue', 'salmon'])
ax.set_xticks(x)
ax.set_xticklabels([f"{k}\n({p:,} params)" for k, p in
                    zip(compare.keys(), params_list)], fontsize=7)
ax.set_ylabel('Test Accuracy')
ax.set_title('Width vs Depth at Similar Parameter Counts')
min_acc = min(accs)
ax.set_ylim(min_acc - 0.005, 1.0)

plt.tight_layout()
plt.savefig('scaling_experiment.png', dpi=150)
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Summary ===")
summary = df.groupby('name').agg(
    params=('n_params', 'first'),
    scaling_type=('scaling_type', 'first'),
    mean_acc=('final_acc', 'mean'),
    std_acc=('final_acc', 'std'),
).sort_values('params')
print(summary.to_string())