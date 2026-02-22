"""
LLM Analysis Practice - Full File
Tests your Vast.ai setup and builds intuition for Q3 of the assessment.
Run with: python llm_analysis.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── Setup ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMALL_MODEL = "Qwen/Qwen2.5-1.5B"
LARGE_MODEL = "Qwen/Qwen2.5-7B"

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory available: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_name):
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"Loaded. GPU memory used so far: {mem:.1f}GB")
    return model, tokenizer

# ── Core functions ─────────────────────────────────────────────────────────────
def get_next_token_dist(model, tokenizer, prompt):
    """Get full next-token probability distribution for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits  = outputs.logits[0, -1, :]
    probs   = torch.softmax(logits, dim=-1).cpu().float()
    logprobs = torch.log_softmax(logits, dim=-1).cpu().float()
    entropy = -(probs * logprobs).sum().item()
    return probs, logprobs, entropy


def top_k_tokens(probs, tokenizer, k=5):
    """Print top k predicted tokens and their probabilities."""
    top = torch.topk(probs, k)
    tokens = [(tokenizer.decode(idx), prob.item())
              for prob, idx in zip(top.values, top.indices)]
    return tokens


def kl(p, q, eps=1e-10):
    """
    KL divergence KL(p || q).
    Measures how much information is lost when q approximates p.
    Asymmetric: kl(p,q) != kl(q,p)
    """
    p = p.float() + eps
    q = q.float() + eps
    return (p * (p / q).log()).sum().item()


def analyze_prompt(small_model, small_tok, large_model, large_tok, prompt, verbose=True):
    """Run full analysis on a single prompt. Returns dict of metrics."""
    p_small, lp_small, ent_small = get_next_token_dist(small_model, small_tok, prompt)
    p_large, lp_large, ent_large = get_next_token_dist(large_model, large_tok, prompt)

    kl_s2l = kl(p_small, p_large)  # how surprised is small by large
    kl_l2s = kl(p_large, p_small)  # how surprised is large by small

    if verbose:
        print(f"\nPrompt: '{prompt}'")
        print(f"  Small model top 5:")
        for token, prob in top_k_tokens(p_small, small_tok):
            print(f"    '{token}': {prob:.4f}")
        print(f"  Large model top 5:")
        for token, prob in top_k_tokens(p_large, large_tok):
            print(f"    '{token}': {prob:.4f}")
        print(f"  Entropy  — small: {ent_small:.3f}  large: {ent_large:.3f}")
        print(f"  KL(small||large): {kl_s2l:.4f}")
        print(f"  KL(large||small): {kl_l2s:.4f}")

    return {
        'prompt':    prompt,
        'ent_small': ent_small,
        'ent_large': ent_large,
        'kl_s2l':    kl_s2l,
        'kl_l2s':    kl_l2s,
        'large_more_confident': ent_large < ent_small,
    }


def get_layer_activation(model, tokenizer, prompt, layer_idx=0):
    """Extract hidden state from a specific transformer layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    activations = {}

    def hook_fn(module, input, output):
        activations['hidden'] = output[0].detach().cpu()

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    # Return last token's activation — this predicts the next token
    return activations['hidden'][0, -1, :]


# ── Prompt sets ───────────────────────────────────────────────────────────────
prompts = {
    'factual': [
        "The capital of France is",
        "The chemical symbol for gold is",
        "The year World War II ended was",
        "The speed of light is approximately",
        "The author of Romeo and Juliet is",
        "The largest planet in our solar system is",
        "Water boils at 100 degrees",
    ],
    'reasoning': [
        "If all cats are mammals and all mammals are animals, then cats are",
        "The next number in the sequence 2, 4, 8, 16 is",
        "If it takes 5 machines 5 minutes to make 5 widgets, 100 machines make widgets in",
        "A bat and ball cost $1.10. The bat costs $1 more than the ball. The ball costs",
        "If today is Monday, in 10 days it will be",
        "The square root of 144 is",
        "If A is greater than B and B is greater than C, then A compared to C is",
    ],
    'ambiguous': [
        "The best programming language is",
        "Pineapple on pizza is",
        "The meaning of life is",
        "The most important quality in a leader is",
        "Dogs are better than cats because",
        "The greatest movie ever made is",
        "The most effective way to learn is",
    ],
}


# ── Main experiment ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load both models
    small_model, small_tok = load_model(SMALL_MODEL)
    large_model, large_tok = load_model(LARGE_MODEL)

    print(f"\nTotal GPU memory used: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── Run analysis across all prompts ───────────────────────────────────────
    print("\n" + "="*60)
    print("RUNNING PROMPT ANALYSIS")
    print("="*60)

    all_results = []
    for category, prompt_list in prompts.items():
        print(f"\n--- Category: {category.upper()} ---")
        for prompt in prompt_list:
            result = analyze_prompt(
                small_model, small_tok,
                large_model, large_tok,
                prompt,
                verbose=True
            )
            result['category'] = category
            all_results.append(result)

    df = pd.DataFrame(all_results)

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY BY CATEGORY")
    print("="*60)

    summary = df.groupby('category').agg(
        mean_kl_s2l=('kl_s2l', 'mean'),
        mean_kl_l2s=('kl_l2s', 'mean'),
        mean_ent_small=('ent_small', 'mean'),
        mean_ent_large=('ent_large', 'mean'),
        pct_large_more_confident=('large_more_confident', 'mean'),
    ).round(4)
    print(summary.to_string())

    print("\n" + "="*60)
    print("KEY QUESTIONS")
    print("="*60)

    # Q1: Which category has highest disagreement?
    top_cat = summary['mean_kl_s2l'].idxmax()
    print(f"\n1. Highest disagreement category: {top_cat}")
    print(f"   KL(small||large) = {summary.loc[top_cat, 'mean_kl_s2l']:.4f}")

    # Q2: Is large model always more confident?
    pct = df['large_more_confident'].mean() * 100
    print(f"\n2. Large model more confident: {pct:.1f}% of prompts")
    print(f"   (not always — depends on prompt type)")

    # Q3: Is KL symmetric?
    mean_s2l = df['kl_s2l'].mean()
    mean_l2s = df['kl_l2s'].mean()
    print(f"\n3. KL symmetry:")
    print(f"   Mean KL(small||large): {mean_s2l:.4f}")
    print(f"   Mean KL(large||small): {mean_l2s:.4f}")
    print(f"   Ratio: {mean_l2s/mean_s2l:.2f}x  (1.0 would be perfectly symmetric)")

    # ── Activation similarity across layers ───────────────────────────────────
    print("\n" + "="*60)
    print("LAYER ACTIVATION ANALYSIS (first layer only)")
    print("="*60)

    # Compare activations on same prompts — note: different hidden sizes
    # so we can't do cosine similarity directly, but we can look at norms
    test_prompts = prompts['factual'][:3]
    print("\nActivation L2 norms (first layer, last token position):")
    for prompt in test_prompts:
        act_small = get_layer_activation(small_model, small_tok, prompt, layer_idx=0)
        act_large = get_layer_activation(large_model, large_tok, prompt, layer_idx=0)
        print(f"  '{prompt[:40]}'")
        print(f"    small norm: {act_small.norm().item():.3f}  shape: {act_small.shape}")
        print(f"    large norm: {act_large.norm().item():.3f}  shape: {act_large.shape}")

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: KL divergence by category
    ax = axes[0, 0]
    categories = list(prompts.keys())
    x = np.arange(len(categories))
    width = 0.35
    s2l_means = [df[df.category == c]['kl_s2l'].mean() for c in categories]
    l2s_means = [df[df.category == c]['kl_l2s'].mean() for c in categories]
    s2l_stds  = [df[df.category == c]['kl_s2l'].std()  for c in categories]
    l2s_stds  = [df[df.category == c]['kl_l2s'].std()  for c in categories]

    ax.bar(x - width/2, s2l_means, width, yerr=s2l_stds,
           label='KL(small||large)', color='steelblue', capsize=4)
    ax.bar(x + width/2, l2s_means, width, yerr=l2s_stds,
           label='KL(large||small)', color='salmon', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence by Prompt Category\n(asymmetry shows which model is more surprised)')
    ax.legend()

    # Plot 2: Entropy comparison
    ax = axes[0, 1]
    ent_small_by_cat = [df[df.category == c]['ent_small'].mean() for c in categories]
    ent_large_by_cat = [df[df.category == c]['ent_large'].mean() for c in categories]
    ent_small_std    = [df[df.category == c]['ent_small'].std()  for c in categories]
    ent_large_std    = [df[df.category == c]['ent_large'].std()  for c in categories]

    ax.bar(x - width/2, ent_small_by_cat, width, yerr=ent_small_std,
           label='Small model', color='steelblue', capsize=4)
    ax.bar(x + width/2, ent_large_by_cat, width, yerr=ent_large_std,
           label='Large model', color='salmon', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('Model Uncertainty by Category\n(higher entropy = more uncertain)')
    ax.legend()

    # Plot 3: KL scatter — is it symmetric per prompt?
    ax = axes[1, 0]
    ax.scatter(df['kl_s2l'], df['kl_l2s'],
               c=[{'factual': 'steelblue', 'reasoning': 'salmon', 'ambiguous': 'green'}[c]
                  for c in df['category']],
               alpha=0.7, s=80)
    max_val = max(df['kl_s2l'].max(), df['kl_l2s'].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Symmetric line')
    ax.set_xlabel('KL(small || large)')
    ax.set_ylabel('KL(large || small)')
    ax.set_title('KL Symmetry per Prompt\n(points above line: large more surprised by small)')
    # Manual legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='factual'),
        Patch(facecolor='salmon',    label='reasoning'),
        Patch(facecolor='green',     label='ambiguous'),
    ]
    ax.legend(handles=legend_elements)

    # Plot 4: Entropy small vs large per prompt
    ax = axes[1, 1]
    ax.scatter(df['ent_small'], df['ent_large'],
               c=[{'factual': 'steelblue', 'reasoning': 'salmon', 'ambiguous': 'green'}[c]
                  for c in df['category']],
               alpha=0.7, s=80)
    max_ent = max(df['ent_small'].max(), df['ent_large'].max()) * 1.1
    ax.plot([0, max_ent], [0, max_ent], 'k--', alpha=0.3, label='Equal confidence')
    ax.set_xlabel('Small model entropy')
    ax.set_ylabel('Large model entropy')
    ax.set_title('Confidence Comparison per Prompt\n(below line: large more confident)')
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig('llm_analysis.png', dpi=150)
    print("\nPlot saved to llm_analysis.png")
    plt.show()

    print("\nDone. If you got here without errors, your Vast.ai setup is working correctly.")