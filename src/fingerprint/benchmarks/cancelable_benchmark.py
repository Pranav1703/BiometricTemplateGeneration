"""
Cancelable Biometric Benchmark

A working benchmark system for cancelable biometrics using cosine similarity matching.

This focuses on what actually works: cancelable transforms with similarity-based verification.

Methods:
1. Raw embeddings (baseline)
2. Cancelable transform (proposed)
3. Salted cancelable (enhanced security)

All methods use cosine similarity for matching.
"""

import numpy as np
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.fingerprint.core.cancelable_transform import CancelableTransform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CancelableBenchmark:
    """
    Benchmark system for cancelable biometric methods.

    Methods:
    1. Raw: Original embeddings (no protection)
    2. Cancelable: Transformed embeddings with user-specific salt
    3. Salted: Enhanced cancelable with additional salt

    Matching: Cosine similarity for all methods
    """

    def __init__(self, embedding_dim: int = 512, cancelable_alpha: float = 0.6):
        """
        Initialize benchmark system.

        Args:
            embedding_dim: Dimension of embeddings
            cancelable_alpha: Cancelable transform parameter
        """
        self.embedding_dim = embedding_dim
        self.cancelable_alpha = cancelable_alpha

        self.cancelable = CancelableTransform(embedding_dim, alpha=cancelable_alpha)

        self.results = {
            "raw": {"genuine_scores": [], "impostor_scores": []},
            "cancelable": {"genuine_scores": [], "impostor_scores": []},
            "cancelable_pbkdf2": {"genuine_scores": [], "impostor_scores": []},
        }

        logger.info("CancelableBenchmark initialized")

    def enroll_raw(
        self, embedding: np.ndarray, user_key: str
    ) -> Tuple[np.ndarray, Dict]:
        """Enroll using raw embedding."""
        embedding = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding, {"user_key": user_key}

    def verify_raw(self, query: np.ndarray, template: Tuple, user_key: str) -> float:
        """Verify using cosine similarity."""
        template_emb, _ = template
        query = query.flatten().astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        return np.dot(template_emb, query)

    def enroll_cancelable(
        self, embedding: np.ndarray, user_key: str
    ) -> Tuple[np.ndarray, Dict]:
        """Enroll using cancelable transform."""
        transformed, params = self.cancelable.enroll(embedding, user_key)
        return transformed, params

    def verify_cancelable(
        self, query: np.ndarray, template: Tuple, user_key: str
    ) -> float:
        """Verify using cancelable transform."""
        template_emb, params = template
        transformed = self.cancelable.verify(query, params)
        return np.dot(template_emb, transformed) / (
            np.linalg.norm(template_emb) * np.linalg.norm(transformed) + 1e-10
        )

    def enroll_pbkdf2(self, embedding: np.ndarray, user_key: str) -> Tuple[Dict, Dict]:
        """Enroll using cancelable transform + PBKDF2 key derivation (Option A)."""
        embedding = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        try:
            template_data, params = self.cancelable.enroll_with_key(embedding, user_key)
            return template_data, params
        except Exception as e:
            logger.warning(f"PBKDF2 enrollment failed: {e}")
            return {"transformed": embedding, "salt": b"", "key_hash": b""}, {
                "user_key": user_key
            }

    def verify_pbkdf2(self, query: np.ndarray, template: Tuple, user_key: str) -> float:
        """Verify using cancelable transform + PBKDF2 (Option A)."""
        template_data, params = template
        if template_data is None:
            return 0.0

        query = query.flatten().astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        try:
            success, _ = self.cancelable.verify_with_key(query, template_data, params)
            return 1.0 if success else 0.0
        except Exception as e:
            logger.warning(f"PBKDF2 verification failed: {e}")
            return 0.0

    def run_tests(
        self,
        embeddings: Dict[str, Dict[str, np.ndarray]],
        num_impostor_tests: int = 100,
    ) -> Dict:
        """
        Run benchmark tests.

        Args:
            embeddings: Dictionary of embeddings per user per hand
            num_impostor_tests: Number of impostor tests to run

        Returns:
            Results dictionary with metrics
        """
        users = list(embeddings.keys())
        all_hands = list(embeddings[users[0]].keys()) if users else []

        logger.info(f"Running benchmark with {len(users)} users, hands: {all_hands}")

        genuine_scores = {"raw": [], "cancelable": [], "cancelable_pbkdf2": []}
        impostor_scores = {"raw": [], "cancelable": [], "cancelable_pbkdf2": []}

        for user_id in tqdm(users, desc="Processing users"):
            user_hands = embeddings[user_id]
            hand_list = list(user_hands.keys())

            for i, hand1 in enumerate(hand_list):
                emb1 = user_hands[hand1]

                templates_raw, templates_cancelable, templates_fuzzy = {}, {}, {}

                try:
                    template_raw, params_raw = self.enroll_raw(emb1, user_id)
                    templates_raw[hand1] = (template_raw, params_raw)

                    template_cancelable, params_cancelable = self.enroll_cancelable(
                        emb1, user_id
                    )
                    templates_cancelable[hand1] = (
                        template_cancelable,
                        params_cancelable,
                    )

                    template_fuzzy, params_fuzzy = self.enroll_pbkdf2(emb1, user_id)
                    templates_fuzzy[hand1] = (template_fuzzy, params_fuzzy)
                except Exception as e:
                    logger.warning(f"Enrollment failed for {user_id}/{hand1}: {e}")
                    continue

                for hand2 in hand_list[i + 1 :]:
                    emb2 = user_hands[hand2]

                    try:
                        score_raw = self.verify_raw(emb2, templates_raw[hand1], user_id)
                        genuine_scores["raw"].append(score_raw)

                        score_cancelable = self.verify_cancelable(
                            emb2, templates_cancelable[hand1], user_id
                        )
                        genuine_scores["cancelable"].append(score_cancelable)

                        score_fuzzy = self.verify_pbkdf2(
                            emb2, templates_fuzzy[hand1], user_id
                        )
                        genuine_scores["cancelable_pbkdf2"].append(score_fuzzy)
                    except Exception as e:
                        logger.warning(f"Genuine verification failed: {e}")

        for _ in range(num_impostor_tests):
            if len(users) < 2:
                break

            user1, user2 = np.random.choice(users, 2, replace=False)

            hand1 = np.random.choice(list(embeddings[user1].keys()))
            hand2 = np.random.choice(list(embeddings[user2].keys()))

            try:
                template1_raw, params1_raw = self.enroll_raw(
                    embeddings[user1][hand1], user1
                )
                score_raw = self.verify_raw(
                    embeddings[user2][hand2], (template1_raw, params1_raw), user2
                )
                impostor_scores["raw"].append(score_raw)

                template1_cancelable, params1_cancelable = self.enroll_cancelable(
                    embeddings[user1][hand1], user1
                )
                score_cancelable = self.verify_cancelable(
                    embeddings[user2][hand2],
                    (template1_cancelable, params1_cancelable),
                    user2,
                )
                impostor_scores["cancelable"].append(score_cancelable)

                template1_fuzzy, params1_fuzzy = self.enroll_pbkdf2(
                    embeddings[user1][hand1], user1
                )
                score_fuzzy = self.verify_pbkdf2(
                    embeddings[user2][hand2],
                    (template1_fuzzy, params1_fuzzy),
                    user2,
                )
                impostor_scores["cancelable_pbkdf2"].append(score_fuzzy)
            except Exception as e:
                logger.warning(f"Impostor test failed: {e}")

        self.results = {
            "raw": genuine_scores,
            "cancelable": genuine_scores,
            "cancelable_pbkdf2": genuine_scores,
        }

        metrics = self.calculate_metrics(genuine_scores, impostor_scores)

        return metrics

    def calculate_metrics(self, genuine: Dict, impostor: Dict) -> Dict:
        """Calculate benchmark metrics."""
        all_genuine = np.concatenate([genuine[k] for k in genuine.keys()])
        all_impostor = np.concatenate([impostor[k] for k in impostor.keys()])

        all_scores = np.concatenate([all_genuine, all_impostor])
        labels = np.concatenate(
            [np.ones(len(all_genuine)), np.zeros(len(all_impostor))]
        )

        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        roc_auc = auc(fpr, tpr)

        eer_threshold_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = (fpr[eer_threshold_idx] + (1 - tpr[eer_threshold_idx])) / 2

        d_prime = (np.mean(all_genuine) - np.mean(all_impostor)) / np.sqrt(
            0.5 * (np.std(all_genuine) ** 2 + np.std(all_impostor) ** 2)
        )

        return {
            "auc": roc_auc,
            "eer": eer,
            "d_prime": d_prime,
            "genuine_mean": np.mean(all_genuine),
            "genuine_std": np.std(all_genuine),
            "impostor_mean": np.mean(all_impostor),
            "impostor_std": np.std(all_impostor),
            "num_genuine": len(all_genuine),
            "num_impostor": len(all_impostor),
        }

    def plot_roc(
        self, output_path: str = "artifacts/cancelable_benchmark/roc_curves.png"
    ):
        """Plot ROC curves."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        for method in ["raw", "cancelable"]:
            if method not in self.results:
                continue
            genuine = self.results[method].get("genuine_scores", [])
            impostor = self.results[method].get("impostor_scores", [])

            if not genuine or not impostor:
                continue

            if not genuine or not impostor:
                continue

            all_scores = np.concatenate([genuine, impostor])
            labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])

            fpr, tpr, _ = roc_curve(labels, all_scores)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f"{method} (AUC = {roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves - Cancelable Biometrics")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"ROC curves saved to {output_path}")

    def print_results(self, metrics: Dict):
        """Print benchmark results."""
        print("\n" + "=" * 60)
        print("CANCELABLE BIOMETRIC BENCHMARK RESULTS")
        print("=" * 60)

        print(f"\nConfiguration:")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Cancelable alpha: {self.cancelable_alpha}")

        print(f"\nMetrics:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  EER: {metrics['eer']:.4f}")
        print(f"  d-prime: {metrics['d_prime']:.4f}")

        print(f"\nGenuine Scores:")
        print(f"  Mean: {metrics['genuine_mean']:.4f}")
        print(f"  Std:  {metrics['genuine_std']:.4f}")

        print(f"\nImpostor Scores:")
        print(f"  Mean: {metrics['impostor_mean']:.4f}")
        print(f"  Std:  {metrics['impostor_std']:.4f}")

        print(f"\nSample Sizes:")
        print(f"  Genuine: {metrics['num_genuine']}")
        print(f"  Impostor: {metrics['num_impostor']}")

        print("=" * 60)


def generate_synthetic_embeddings(
    num_users: int = 50,
    embedding_dim: int = 512,
    intra_class_std: float = 0.02,
    inter_class_std: float = 0.5,
) -> Dict:
    """
    Generate synthetic biometric embeddings.

    Args:
        num_users: Number of users
        embedding_dim: Embedding dimension
        intra_class_std: Standard deviation within same user (noise)
        inter_class_std: Standard deviation between users (variability)

    Returns:
        Dictionary of embeddings per user per hand
    """
    embeddings = {}

    for i in range(num_users):
        user_id = f"user_{i:03d}"

        user_mean = np.random.randn(embedding_dim).astype(np.float32)
        user_mean = user_mean / np.linalg.norm(user_mean) * inter_class_std

        embeddings[user_id] = {}

        for hand in ["L", "R"]:
            noise = np.random.randn(embedding_dim).astype(np.float32) * intra_class_std
            embedding = user_mean + noise
            embedding = embedding / np.linalg.norm(embedding)
            embeddings[user_id][hand] = embedding

    return embeddings


def run_synthetic_benchmark():
    """Run benchmark with synthetic data."""
    print("\n" + "=" * 60)
    print("CANCELABLE BIOMETRIC BENCHMARK")
    print("=" * 60)

    print("\nGenerating synthetic embeddings...")
    embeddings = generate_synthetic_embeddings(
        num_users=50, embedding_dim=512, intra_class_std=0.02, inter_class_std=0.5
    )

    print(f"Generated {len(embeddings)} users")

    benchmark = CancelableBenchmark(embedding_dim=512, cancelable_alpha=0.6)

    print("\nRunning benchmark...")
    metrics = benchmark.run_tests(embeddings, num_impostor_tests=200)

    benchmark.print_results(metrics)
    benchmark.plot_roc()

    return metrics


def load_real_embeddings(
    embeddings_path: str = "artifacts/embedding/fvc2000_test_embeddings.pt",
) -> Dict:
    """Load real embeddings from file."""
    print(f"\nLoading embeddings from {embeddings_path}...")
    data = torch.load(embeddings_path)
    embeddings_tensor = data["embeddings"]
    labels = data["labels"]

    embeddings_dict = {}

    for idx, label in enumerate(labels):
        user_id = str(label.item())
        embedding = embeddings_tensor[idx].numpy()

        if user_id not in embeddings_dict:
            embeddings_dict[user_id] = {}

        hand_idx = len(embeddings_dict[user_id])
        hand_key = f"sample_{hand_idx}"
        embeddings_dict[user_id][hand_key] = embedding.astype(np.float32)

    print(f"Loaded embeddings for {len(embeddings_dict)} users")
    return embeddings_dict


def run_real_benchmark(
    embeddings_path: str = "artifacts/embedding/fvc2000_test_embeddings.pt",
):
    """Run benchmark with real FVC2000 embeddings."""
    print("\n" + "=" * 60)
    print("CANCELABLE BIOMETRIC BENCHMARK - FVC2000")
    print("=" * 60)

    embeddings = load_real_embeddings(embeddings_path)

    benchmark = CancelableBenchmark(embedding_dim=512, cancelable_alpha=0.6)

    print("\nRunning benchmark...")
    metrics = benchmark.run_tests(embeddings, num_impostor_tests=200)

    benchmark.print_results(metrics)
    benchmark.plot_roc("artifacts/plots/cancelable_benchmark_roc.png")

    return metrics


if __name__ == "__main__":
    run_real_benchmark()
