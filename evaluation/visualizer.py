"""
Publication-Quality Biometric Visualizer Module.
Generates research-grade figures for the Master's Thesis:
  1. ROC Curves (Receiver Operating Characteristic) & EER Operating Points
  2. FAR vs. FRR Trade-off Curves
  3. 2D t-SNE Latent Space Projections (Baseline vs Proposed Contrastive)
  4. Spoof-Resistance & Replay Attack Defense Evaluation Charts
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for automated rendering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, Any, List

class BiometricVisualizer:
    """
    Renders high-resolution scientific plots saved in results/figures/.
    """
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Configure publication styling
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

    def plot_roc_curves(self, models_results: Dict[str, Dict[str, Any]], filename: str = "roc_curves.png") -> str:
        """
        Plots multi-model ROC Curves with AUC and EER annotations.
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for idx, (model_name, res) in enumerate(models_results.items()):
            curves = res.get("_curves", {})
            fpr = curves.get("fpr", np.array([0, 1]))
            tpr = curves.get("tpr", np.array([0, 1]))
            auc_val = res.get("AUC", 0.5)
            eer_val = res.get("EER", 0.5)

            color = colors[idx % len(colors)]
            ax.plot(
                fpr, tpr,
                label=f"{model_name} (AUC={auc_val*100:.1f}%, EER={eer_val*100:.1f}%)",
                linewidth=2.0,
                color=color
            )

        # Diagonal chance line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC=50%)')

        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel('False Acceptance Rate (FAR / FPR)')
        ax.set_ylabel('True Acceptance Rate (TAR = 1 - FRR)')
        ax.set_title('Cross-Session Biometric Verification Performance (ROC)')
        ax.legend(loc='lower right', frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)

        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_far_frr_tradeoff(self, evaluation_res: Dict[str, Any], model_name: str, filename: str = "far_frr_tradeoff.png") -> str:
        """
        Plots FAR and FRR vs. Similarity Threshold, highlighting the EER crossover.
        """
        curves = evaluation_res["_curves"]
        thresholds = curves["thresholds"]
        far = curves["far"]
        frr = curves["frr"]
        eer = evaluation_res["EER"]
        eer_th = evaluation_res["EER_threshold"]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(thresholds, far * 100, label='False Acceptance Rate (FAR)', color='#d62728', linewidth=2.0)
        ax.plot(thresholds, frr * 100, label='False Rejection Rate (FRR)', color='#1f77b4', linewidth=2.0)

        # Highlight EER
        ax.scatter([eer_th], [eer * 100], color='black', zorder=5, s=60)
        ax.axvline(x=eer_th, color='gray', linestyle='--', alpha=0.7)
        ax.axhline(y=eer * 100, color='gray', linestyle='--', alpha=0.7)

        ax.annotate(
            f'EER = {eer*100:.2f}%\nThreshold = {eer_th:.3f}',
            xy=(eer_th, eer * 100),
            xytext=(eer_th + 0.1, eer * 100 + 10),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([-1, 101])
        ax.set_xlabel('Cosine Similarity Decision Threshold')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title(f'FAR vs. FRR Operational Trade-off — {model_name}')
        ax.legend(loc='center right', frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)

        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_tsne_comparison(
        self,
        baseline_embeddings: np.ndarray,
        contrastive_embeddings: np.ndarray,
        subject_ids: np.ndarray,
        session_ids: np.ndarray,
        max_subjects: int = 6,
        filename: str = "latent_space_tsne.png"
    ) -> str:
        """
        Side-by-side t-SNE visualization comparing:
          Left: Standard Cross-Entropy Latent Space (fragmented sessions)
          Right: Supervised Contrastive Latent Space (compact subject clusters across sessions)
        """
        # Filter to subset of subjects for visual clarity
        mask = subject_ids < max_subjects
        sub_ids = subject_ids[mask]
        ses_ids = session_ids[mask]

        b_embs = baseline_embeddings[mask]
        c_embs = contrastive_embeddings[mask]

        tsne = TSNE(n_components=2, perplexity=25, random_state=42)
        b_2d = tsne.fit_transform(b_embs)
        c_2d = tsne.fit_transform(c_embs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        cmap = plt.get_cmap('tab10')

        markers = ['o', 's', '^', 'D', 'v', 'p']

        for sub in range(max_subjects):
            # Left Plot: Baseline
            sub_mask = (sub_ids == sub)
            for ses in np.unique(ses_ids):
                idx = sub_mask & (ses_ids == ses)
                if np.any(idx):
                    ax1.scatter(
                        b_2d[idx, 0], b_2d[idx, 1],
                        color=cmap(sub),
                        marker=markers[ses % len(markers)],
                        alpha=0.7,
                        s=35,
                        label=f"Sub {sub+1}" if ses == 0 else None
                    )

            # Right Plot: Contrastive
            for ses in np.unique(ses_ids):
                idx = sub_mask & (ses_ids == ses)
                if np.any(idx):
                    ax2.scatter(
                        c_2d[idx, 0], c_2d[idx, 1],
                        color=cmap(sub),
                        marker=markers[ses % len(markers)],
                        alpha=0.7,
                        s=35,
                        label=f"Subject {sub+1}" if ses == np.unique(ses_ids)[0] else None
                    )

        ax1.set_title("(a) Standard Cross-Entropy Latent Space\n(Noticeable session drift & overlap)", fontsize=11)
        ax1.set_xlabel("t-SNE Dimension 1")
        ax1.set_ylabel("t-SNE Dimension 2")
        ax1.grid(True, linestyle=':', alpha=0.4)

        ax2.set_title("(b) Supervised Contrastive Latent Space (Ours)\n(Tight intra-subject clustering across sessions)", fontsize=11)
        ax2.set_xlabel("t-SNE Dimension 1")
        ax2.set_ylabel("t-SNE Dimension 2")
        ax2.legend(title="Subjects", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, linestyle=':', alpha=0.4)

        plt.suptitle("Latent Feature Space Manifold Comparison Across Recording Sessions", fontsize=13, y=1.02)

        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_replay_attack_resilience(
        self,
        attack_results: Dict[str, Dict[str, float]],
        filename: str = "replay_attack_evaluation.png"
    ) -> str:
        """
        Bar chart comparing acceptance rates under attack scenarios:
          Without Liveness Defense vs. With Proposed Liveness Defense.
        """
        attack_types = list(attack_results.keys())
        no_defense = [attack_results[a]["without_defense"] * 100 for a in attack_types]
        with_defense = [attack_results[a]["with_defense"] * 100 for a in attack_types]

        x = np.arange(len(attack_types))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        rects1 = ax.bar(x - width/2, no_defense, width, label='Static Biometric Gate (No Liveness)', color='#d62728', alpha=0.85)
        rects2 = ax.bar(x + width/2, with_defense, width, label='Proposed Spoof-Resistant Framework', color='#2ca02c', alpha=0.85)

        ax.set_ylabel('Access Acceptance Rate (%)')
        ax.set_title('System Resilience Under Simulated Presentation & Replay Attacks')
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('_', ' ').title() for a in attack_types], rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.set_ylim([0, 115])

        # Add value labels on bars
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        return save_path
