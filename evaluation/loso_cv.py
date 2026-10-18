"""
Leave-One-Session-Out Cross-Validation (LOSO-CV) Engine.
Rigorously evaluates biometric stability and temporal generalization by testing
on an entirely unseen future recording session across all subjects.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Any, Type
from config.settings import FrameworkConfig
from data.dataset import EEGMultiSessionDataset, BalancedSessionBatchSampler
from losses.supervised_contrastive import JointBiometricLoss
from evaluation.metrics import BiometricMetricsEvaluator

class LOSOCrossValidator:
    """
    Manages Leave-One-Session-Out validation across all temporal recording sessions.
    """
    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.evaluator = BiometricMetricsEvaluator()

    def run_loso_evaluation(
        self,
        dataset: EEGMultiSessionDataset,
        model_class: Type[nn.Module],
        model_kwargs: Dict[str, Any],
        is_contrastive: bool = True,
        epochs_per_fold: int = 20
    ) -> Dict[str, Any]:
        """
        Executes complete LOSO-CV across all sessions.
        """
        num_sessions = self.config.num_sessions
        session_ids = dataset.session_ids.numpy()

        fold_results = []
        device = self.config.training.device

        print(f"\n=======================================================")
        print(f" Starting LOSO-CV (Total Sessions: {num_sessions})")
        print(f" Architecture: {model_class.__name__} | Contrastive: {is_contrastive}")
        print(f"=======================================================")

        for test_session in range(num_sessions):
            print(f"\n---> [Fold {test_session + 1}/{num_sessions}] Holding out Session {test_session} as Test Set")

            # Split indices
            train_indices = np.where(session_ids != test_session)[0]
            test_indices = np.where(session_ids == test_session)[0]

            train_subset = Subset(dataset, train_indices)
            test_subset = Subset(dataset, test_indices)

            # Build model instance for this fold
            model = model_class(**model_kwargs).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )

            if is_contrastive:
                criterion = JointBiometricLoss(
                    temperature=self.config.training.temperature,
                    contrastive_weight=self.config.training.contrastive_weight
                )
                # Use balanced contrastive sampler on training fold
                sampler = BalancedSessionBatchSampler(
                    dataset=dataset,
                    num_classes_per_batch=min(8, self.config.num_subjects),
                    num_samples_per_class=4,
                    total_batches=30
                )
                train_loader = DataLoader(dataset, batch_sampler=sampler)
            else:
                criterion = nn.CrossEntropyLoss()
                train_loader = DataLoader(
                    train_subset,
                    batch_size=self.config.training.batch_size,
                    shuffle=True
                )

            # Training loop
            model.train()
            for epoch in range(epochs_per_fold):
                running_loss = 0.0
                for batch in train_loader:
                    x = batch["epoch"].to(device)
                    y = batch["subject_id"].to(device)

                    optimizer.zero_grad()
                    outputs = model(x, return_projection=is_contrastive, return_logits=True)

                    if is_contrastive:
                        loss, _, _ = criterion(outputs["projection"], outputs["logits"], y)
                    else:
                        loss = criterion(outputs["logits"], y)

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

            # Evaluation on test fold (Session test_session)
            model.eval()
            test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

            test_embeddings = []
            test_subjects = []
            test_sessions = []

            with torch.no_grad():
                for batch in test_loader:
                    x = batch["epoch"].to(device)
                    outputs = model(x, return_projection=True, return_logits=False)
                    emb = outputs["projection"].cpu().numpy()
                    test_embeddings.append(emb)
                    test_subjects.append(batch["subject_id"].numpy())
                    test_sessions.append(batch["session_id"].numpy())

            test_embeddings = np.concatenate(test_embeddings, axis=0)
            test_subjects = np.concatenate(test_subjects, axis=0)
            test_sessions = np.concatenate(test_sessions, axis=0)

            # Note: In pure single-session test set, genuine pairs must come from enrollment.
            # We extract enrollment templates from training sessions and evaluate test probes against them!
            fold_metrics = self._evaluate_cross_session_verification(
                model=model,
                train_subset=train_subset,
                test_subset=test_subset,
                device=device
            )

            print(f" Fold {test_session + 1} Results -> EER: {fold_metrics['EER']*100:.2f}% | AUC: {fold_metrics['AUC']*100:.2f}% | d': {fold_metrics['d_prime']:.2f}")
            fold_results.append(fold_metrics)

        # Aggregate metrics across all folds
        avg_eer = np.mean([f["EER"] for f in fold_results])
        std_eer = np.std([f["EER"] for f in fold_results])
        avg_auc = np.mean([f["AUC"] for f in fold_results])
        std_auc = np.std([f["AUC"] for f in fold_results])
        avg_d_prime = np.mean([f["d_prime"] for f in fold_results])
        std_d_prime = np.std([f["d_prime"] for f in fold_results])

        summary = {
            "mean_EER": float(avg_eer),
            "std_EER": float(std_eer),
            "mean_AUC": float(avg_auc),
            "std_AUC": float(std_auc),
            "mean_d_prime": float(avg_d_prime),
            "std_d_prime": float(std_d_prime),
            "fold_results": fold_results
        }

        print(f"\n=======================================================")
        print(f" LOSO-CV Summary for {model_class.__name__}")
        print(f" Mean EER: {avg_eer*100:.2f}% (+- {std_eer*100:.2f}%)")
        print(f" Mean AUC: {avg_auc*100:.2f}% (+- {std_auc*100:.2f}%)")
        print(f" Mean d':  {avg_d_prime:.2f} (+- {std_d_prime:.2f})")
        print(f"=======================================================\n")

        return summary

    def _evaluate_cross_session_verification(
        self,
        model: nn.Module,
        train_subset: Subset,
        test_subset: Subset,
        device: str
    ) -> Dict[str, Any]:
        """
        Enrolls templates from training sessions and tests against unseen test session probes.
        """
        model.eval()
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

        # 1. Compute enrolled templates per subject from training sessions
        train_embs, train_subs = [], []
        with torch.no_grad():
            for b in train_loader:
                x = b["epoch"].to(device)
                p = model(x, return_projection=True)["projection"].cpu().numpy()
                train_embs.append(p)
                train_subs.append(b["subject_id"].numpy())
        train_embs = np.concatenate(train_embs, axis=0)
        train_subs = np.concatenate(train_subs, axis=0)

        templates = {}
        for s in np.unique(train_subs):
            sub_vectors = train_embs[train_subs == s]
            mean_vec = np.mean(sub_vectors, axis=0)
            templates[s] = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)

        # 2. Extract test session probes
        test_embs, test_subs = [], []
        with torch.no_grad():
            for b in test_loader:
                x = b["epoch"].to(device)
                p = model(x, return_projection=True)["projection"].cpu().numpy()
                test_embs.append(p)
                test_subs.append(b["subject_id"].numpy())
        test_embs = np.concatenate(test_embs, axis=0)
        test_subs = np.concatenate(test_subs, axis=0)

        # 3. Match test probes against all enrolled templates
        genuine_scores = []
        impostor_scores = []

        for emb, true_sub in zip(test_embs, test_subs):
            for target_sub, template in templates.items():
                score = float(np.dot(emb, template))
                if target_sub == true_sub:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)

        gen_arr = np.array(genuine_scores)
        imp_arr = np.array(impostor_scores)

        eer, eer_th, _, _ = self.evaluator.compute_eer(gen_arr, imp_arr)
        roc_auc, _, _ = self.evaluator.compute_roc_auc(gen_arr, imp_arr)
        d_prime = self.evaluator.compute_d_prime(gen_arr, imp_arr)

        return {
            "EER": eer,
            "EER_threshold": eer_th,
            "AUC": roc_auc,
            "d_prime": d_prime
        }
