import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from env.workload_management import workload


def plot_scenarios(results_dir="results"):
    """Plot incoming requests and forwarding capacity profiles for Scenarios 1-3."""
    steps = np.arange(100)
    period = 50
    average_requests = 100
    amplitude_requests = 50

    s1_req, s1_fwd = zip(*[workload.scenario1(average_requests=average_requests) for _ in steps])
    s2_req, s2_fwd = zip(*[workload.scenario2(t, period, average_requests=average_requests, amplitude_requests=amplitude_requests) for t in steps])
    s3_req, s3_fwd = zip(*[workload.scenario3(average_requests, amplitude_requests, t, period) for t in steps])

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
    scenarios = [
        ("Scenario 1 (Random / Gaussian)", s1_req, s1_fwd),
        ("Scenario 2 (Noisy Sinusoid)", s2_req, s2_fwd),
        ("Scenario 3 (Deterministic Sinusoid)", s3_req, s3_fwd),
    ]

    for ax, (title, req, fwd) in zip(axes, scenarios):
        ax.plot(steps, req, label="Incoming Requests", linewidth=1.8)
        ax.plot(steps, fwd, label="Forwarding Capacity", linestyle="--", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Requests / Capacity")
    fig.suptitle("Simulation Workload Scenarios", fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, "scenarios_overview.png"), dpi=150)
    plt.close(fig)


def plot_algorithm_comparison(results_dir="results"):
    """Compare PPO, SAC, and NEAT evaluation metrics from saved CSV results."""
    algorithms = {
        "PPO": os.path.join("logs", "PPO", "results.csv"),
        "SAC": os.path.join("logs", "SAC", "results.csv"),
        "NEAT": os.path.join("logs", "NEAT", "results.csv"),
    }

    dfs = {}
    for name, path in algorithms.items():
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)

    if not dfs:
        raise FileNotFoundError("No evaluation results CSV files found in logs/.")

    names = list(dfs.keys())
    colors = ["#2b7bba", "#e67e22", "#27ae60"][:len(names)]

    mean_rewards = [dfs[n]["Ricompensa Totale"].mean() for n in names]
    std_rewards = [dfs[n]["Ricompensa Totale"].std() for n in names]
    mean_rej_pct = [dfs[n]["Percentuale di Richieste Rifiutate"].mean() for n in names]
    mean_cong = [dfs[n]["Steps in Congestione"].mean() for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].bar(names, mean_rewards, yerr=std_rewards, capsize=6, color=colors, alpha=0.88)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Mean Total Reward (50 Episodes)")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(names, mean_rej_pct, color=colors, alpha=0.88)
    axes[1].set_title("Mean Rejection Rate (%)")
    axes[1].set_ylabel("Rejected Requests (%)")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(names, mean_cong, color=colors, alpha=0.88)
    axes[2].set_title("Mean Congestion Steps per Episode")
    axes[2].set_ylabel("Congestion Steps")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Algorithm Evaluation Comparison (Scenario 1, 50 Episodes)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, "algorithm_comparison.png"), dpi=150)
    plt.close(fig)

    summary = []
    for n in names:
        df = dfs[n]
        summary.append({
            "Algorithm": n,
            "Mean Reward": df["Ricompensa Totale"].mean(),
            "Max Reward": df["Ricompensa Totale"].max(),
            "Mean Rejection (%)": df["Percentuale di Richieste Rifiutate"].mean(),
            "Mean Congestion Steps": df["Steps in Congestione"].mean(),
        })
    return pd.DataFrame(summary)


if __name__ == "__main__":
    plot_scenarios()
    summary_df = plot_algorithm_comparison()
    print(summary_df.to_string(index=False))

