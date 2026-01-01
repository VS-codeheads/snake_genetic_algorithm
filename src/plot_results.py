"""
Generate visualizations from training results JSON files.
Creates learning curves and comparison plots for the paper.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'feature': '#1f77b4', 'cnn': '#ff7f0e', 'random': '#2ca02c', 'human': '#d62728'}

def load_results(agent_type, results_dir='results'):
    """Load training results for an agent type."""
    agent_dir = f"{results_dir}/dqn_{agent_type}"
    results = []
    
    for seed in [42, 123, 456]:
        # Try both naming conventions
        path1 = f"{agent_dir}/dqn_{agent_type}_seed{seed}_results.json"
        path2 = f"{agent_dir}/results_seed{seed}.json"
        
        if os.path.exists(path1):
            with open(path1, 'r') as f:
                results.append(json.load(f))
        elif os.path.exists(path2):
            with open(path2, 'r') as f:
                results.append(json.load(f))
    
    return results


def plot_learning_curves(results_feature, results_cnn, output_path='report/figures/learning_curves.png'):
    """Plot learning curves for feature and CNN agents."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training scores over episodes
    ax = axes[0]
    for result in results_feature:
        seed = result['seed']
        scores = result['episode_scores']
        # Smooth with rolling average
        smoothed = np.convolve(scores, np.ones(50)/50, mode='valid')
        ax.plot(range(len(smoothed)), smoothed, label=f"Feature (seed {seed})", 
                color=colors['feature'], linewidth=2, alpha=0.7)
    
    for result in results_cnn:
        seed = result['seed']
        scores = result['episode_scores']
        smoothed = np.convolve(scores, np.ones(50)/50, mode='valid')
        ax.plot(range(len(smoothed)), smoothed, label=f"CNN (seed {seed})", 
                color=colors['cnn'], linewidth=2, alpha=0.7, linestyle='--')
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (50-episode MA)', fontsize=12, fontweight='bold')
    ax.set_title('Training Progress', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation metrics over time
    ax = axes[1]
    for result in results_feature:
        seed = result['seed']
        evals = result['evaluations']
        episodes = [e['episode'] for e in evals]
        scores = [e['mean_score'] for e in evals]
        ax.plot(episodes, scores, marker='o', label=f"Feature (seed {seed})", 
                color=colors['feature'], linewidth=2, markersize=8)
    
    for result in results_cnn:
        seed = result['seed']
        evals = result['evaluations']
        episodes = [e['episode'] for e in evals]
        scores = [e['mean_score'] for e in evals]
        ax.plot(episodes, scores, marker='s', label=f"CNN (seed {seed})", 
                color=colors['cnn'], linewidth=2, markersize=8, linestyle='--')
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Evaluation Score (ε=0)', fontsize=12, fontweight='bold')
    ax.set_title('Evaluation Performance', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curves to {output_path}")
    plt.close()


def plot_comparison(results_feature, results_cnn, human_score=2.5, output_path='report/figures/comparison.png'):
    """Plot comparison bar chart."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract final evaluation scores
    feature_scores = [r['evaluations'][-1]['mean_score'] for r in results_feature]
    feature_std = [r['evaluations'][-1]['std_score'] for r in results_feature]
    
    cnn_scores = [r['evaluations'][-1]['mean_score'] for r in results_cnn]
    cnn_std = [r['evaluations'][-1]['std_score'] for r in results_cnn]
    
    # Plot 1: Mean score comparison
    ax = axes[0]
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, feature_scores, width, label='DQN-Feature', 
           color=colors['feature'], yerr=feature_std, capsize=5, alpha=0.8)
    ax.bar(x + width/2, cnn_scores, width, label='DQN-CNN', 
           color=colors['cnn'], yerr=cnn_std, capsize=5, alpha=0.8)
    ax.axhline(y=human_score, color=colors['human'], linestyle='--', linewidth=2, label='Human')
    ax.axhline(y=0.1, color=colors['random'], linestyle='--', linewidth=2, label='Random')
    
    ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Random Seed', fontsize=12, fontweight='bold')
    ax.set_title('Final Evaluation Score by Seed', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['42', '123', '456'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Aggregate statistics
    ax = axes[1]
    agents = ['DQN-Feature', 'DQN-CNN', 'Human', 'Random']
    means = [
        np.mean(feature_scores),
        np.mean(cnn_scores),
        human_score,
        0.1
    ]
    stds = [
        np.mean(feature_std),
        np.mean(cnn_std),
        0,
        0
    ]
    colors_list = [colors['feature'], colors['cnn'], colors['human'], colors['random']]
    
    bars = ax.bar(agents, means, yerr=stds, capsize=8, color=colors_list, alpha=0.8)
    ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax.set_title('Aggregate Performance Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_survival_time(results_feature, results_cnn, human_survival=1100, output_path='report/figures/survival.png'):
    """Plot survival time (steps per episode) comparison."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract survival times
    feature_survival = [r['evaluations'][-1]['mean_survival'] for r in results_feature]
    cnn_survival = [r['evaluations'][-1]['mean_survival'] for r in results_cnn]
    
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, feature_survival, width, label='DQN-Feature', 
           color=colors['feature'], alpha=0.8)
    ax.bar(x + width/2, cnn_survival, width, label='DQN-CNN', 
           color=colors['cnn'], alpha=0.8)
    ax.axhline(y=human_survival, color=colors['human'], linestyle='--', linewidth=2, label='Human')
    ax.axhline(y=380, color=colors['random'], linestyle='--', linewidth=2, label='Random')
    
    ax.set_ylabel('Mean Survival Time (steps)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Random Seed', fontsize=12, fontweight='bold')
    ax.set_title('Mean Episode Length (ε=0)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['42', '123', '456'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1200])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved survival time plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Loading training results...")
    results_feature = load_results('feature')
    results_cnn = load_results('cnn')
    
    if not results_feature:
        print("Warning: No feature results found. Skipping plots.")
    else:
        print(f"Found {len(results_feature)} feature agent runs")
    
    if not results_cnn:
        print("Warning: No CNN results found. Skipping plots.")
    else:
        print(f"Found {len(results_cnn)} CNN agent runs")
    
    if results_feature and results_cnn:
        print("\nGenerating figures...")
        plot_learning_curves(results_feature, results_cnn)
        plot_comparison(results_feature, results_cnn, human_score=2.5)
        plot_survival_time(results_feature, results_cnn, human_survival=1100)
        print("\nAll figures generated successfully!")
    else:
        print("\nSkipping figure generation due to missing results.")
