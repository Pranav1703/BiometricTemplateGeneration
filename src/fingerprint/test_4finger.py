"""
4-Finger Biometric Authentication Test

Tests the complete architecture:
- Raw Embedding → Cancelable Transform → Transformed Embedding
- Path A: Cosine Similarity → Authentication
- Path B: PBKDF2 → Key Derivation

Metrics calculated:
- FAR (False Accept Rate): Impostor accepted as genuine
- FVR (False Reject Rate): Genuine rejected as impostor
- EER (Equal Error Rate): FAR = FVR
- GAR (Genuine Accept Rate): 1 - FVR
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FVC2000_DB1A_DIR,
    FVC2004_DIR,
    FVC2000_ENROLLMENT_FINGERS,
    FVC2000_VERIFICATION_FINGERS,
    FVC2000_NUM_PERSONS,
    FVC2004_ENROLLMENT_FINGERS,
    FVC2004_VERIFICATION_FINGERS,
    FVC2004_NUM_PERSONS,
    DEFAULT_CANCELABLE_ALPHA,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from src.fingerprint.core.cancelable_transform import CancelableTransform


# Dataset configuration mapping
DATASET_CONFIGS = {
    "fvc2000": {
        "data_dir": FVC2000_DB1A_DIR,
        "enrollment_fingers": FVC2000_ENROLLMENT_FINGERS,
        "verification_fingers": FVC2000_VERIFICATION_FINGERS,
        "num_persons": FVC2000_NUM_PERSONS,
    },
    "fvc2004": {
        "data_dir": FVC2004_DIR,
        "enrollment_fingers": FVC2004_ENROLLMENT_FINGERS,
        "verification_fingers": FVC2004_VERIFICATION_FINGERS,
        "num_persons": FVC2004_NUM_PERSONS,
    },
}


def get_dataset_config(dataset: str) -> Dict:
    """Get configuration for specified dataset."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Choose from: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset]


def load_fvc2000_4finger_data(
    data_dir: str = FVC2000_DB1A_DIR,
    enrollment_fingers: List[int] = FVC2000_ENROLLMENT_FINGERS,
    verification_fingers: List[int] = FVC2000_VERIFICATION_FINGERS,
    num_persons: int = FVC2000_NUM_PERSONS,
) -> Dict:
    """
    Load FVC2000 4-finger dataset.

    Returns:
        Dictionary with:
        - enrollment: {person_id: {finger_id: embedding}}
        - verification: {person_id: {finger_id: embedding}}
    """
    enrollment_data = {}
    verification_data = {}

    for person_id in range(1, num_persons + 1):
        # Load enrollment fingers (e.g., 1, 2)
        enrollment_data[person_id] = {}
        for finger_id in enrollment_fingers:
            filename = f"{person_id}_{finger_id}.tif"
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                enrollment_data[person_id][finger_id] = filepath

        # Load verification fingers (e.g., 3, 4)
        verification_data[person_id] = {}
        for finger_id in verification_fingers:
            filename = f"{person_id}_{finger_id}.tif"
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                verification_data[person_id][finger_id] = filepath

    return {"enrollment": enrollment_data, "verification": verification_data}


def generate_synthetic_embeddings(
    num_persons: int = 100,
    embedding_dim: int = 512,
    fingers_per_person: int = 2,
    intra_class_std: float = 0.02,
    inter_class_std: float = 0.5,
) -> Dict:
    """
    Generate synthetic embeddings for testing when real data is not available.

    Args:
        num_persons: Number of different persons
        embedding_dim: Dimension of embeddings
        fingers_per_person: Number of fingers per person
        intra_class_std: Variation within same person
        inter_class_std: Variation between different persons

    Returns:
        Dictionary with enrollment and verification embeddings
    """
    enrollment_data = {}
    verification_data = {}

    for person_id in range(num_persons):
        # Each person has a unique "mean" embedding
        person_mean = np.random.randn(embedding_dim).astype(np.float32)
        person_mean = person_mean / np.linalg.norm(person_mean) * inter_class_std

        # Generate enrollment finger(s)
        enrollment_data[person_id] = {}
        for _ in range(fingers_per_person):
            noise = np.random.randn(embedding_dim).astype(np.float32) * intra_class_std
            embedding = person_mean + noise
            embedding = embedding / np.linalg.norm(embedding)
            finger_id = len(enrollment_data[person_id]) + 1
            enrollment_data[person_id][finger_id] = embedding

        # Generate verification finger(s) - slightly more variation
        verification_data[person_id] = {}
        for _ in range(fingers_per_person):
            noise = np.random.randn(embedding_dim).astype(np.float32) * (
                intra_class_std * 1.5
            )
            embedding = person_mean + noise
            embedding = embedding / np.linalg.norm(embedding)
            finger_id = len(verification_data[person_id]) + 1
            verification_data[person_id][finger_id] = embedding

    return {"enrollment": enrollment_data, "verification": verification_data}


class BiometricAuthenticator:
    """
    Complete biometric authentication system implementing the architecture:

    Raw Embedding (512-dim)
            │
            ▼
    Cancelable Transform R = α·R_bio + (1-α)·R_key
            │
            ▼
    Transformed Embedding
            │
            ├─────────────────────┐
            ▼                     ▼
    PATH A:                 PATH B:
    Cosine Similarity      Stable Bits → PBKDF2 → Key
            │                     │
            ▼                     ▼
    Authentication        Cryptographic Key
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        alpha: float = DEFAULT_CANCELABLE_ALPHA,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold
        self.cancelable = CancelableTransform(embedding_dim, alpha=alpha)

    def enroll(self, embedding: np.ndarray, user_key: str) -> Tuple[Dict, Dict]:
        """
        Enroll a biometric template.

        Returns:
            (template, params)
            - template: {transformed, salt, key_hash}
            - params: {user_key, alpha, embedding_dim}
        """
        template, params = self.cancelable.enroll_with_key(embedding, user_key)
        return template, params

    def verify(
        self,
        query_embedding: np.ndarray,
        template: Dict,
        params: Dict,
    ) -> Tuple[bool, float, bytes]:
        """
        Verify a biometric probe against enrolled template.

        Returns:
            (success, similarity, key)
            - success: True if authenticated
            - similarity: Cosine similarity score
            - key: Derived cryptographic key (if success)
        """
        # Apply cancelable transform
        transformed_query = self.cancelable.verify(query_embedding, params)
        transformed_enrolled = template["transformed"]

        # Path A: Cosine similarity for authentication
        similarity = np.dot(transformed_query, transformed_enrolled) / (
            np.linalg.norm(transformed_query) * np.linalg.norm(transformed_enrolled)
            + 1e-10
        )

        # Authentication decision
        success = similarity >= self.similarity_threshold

        # Path B: Derive key (only if authenticated)
        key = b""
        if success:
            salt = template["salt"]
            _, key = self.cancelable.derive_key(transformed_query, salt, key_length=32)

        return success, similarity, key


def compute_metrics(
    genuine_scores: List[float],
    impostor_scores: List[float],
) -> Dict:
    """
    Compute authentication metrics.

    Args:
        genuine_scores: Similarity scores for genuine matches
        impostor_scores: Similarity scores for impostor matches

    Returns:
        Dictionary with FAR, FVR, EER, GAR, etc.
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Compute FAR at different thresholds
    thresholds = np.linspace(-1, 1, 1000)
    far_list = []
    fvr_list = []

    for threshold in thresholds:
        # FAR: Impostor accepted (impostor score >= threshold)
        far = np.mean(impostor_scores >= threshold)
        # FVR: Genuine rejected (genuine score < threshold)
        fvr = np.mean(genuine_scores < threshold)
        far_list.append(far)
        fvr_list.append(fvr)

    far_list = np.array(far_list)
    fvr_list = np.array(fvr_list)

    # Find EER (where FAR = FVR)
    eer_idx = np.argmin(np.abs(far_list - fvr_list))
    eer = (far_list[eer_idx] + fvr_list[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    # Compute AUC using sklearn
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate(
        [np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))]
    )
    fpr, tpr, _ = roc_curve(labels, all_scores)
    roc_auc = auc(fpr, tpr)

    # GAR at specific FAR levels
    gar_at_far_01 = tpr[np.where(fpr <= 0.01)[0][0]] if np.any(fpr <= 0.01) else 0
    gar_at_far_001 = tpr[np.where(fpr <= 0.001)[0][0]] if np.any(fpr <= 0.001) else 0

    # d-prime
    d_prime = (np.mean(genuine_scores) - np.mean(impostor_scores)) / np.sqrt(
        0.5 * (np.std(genuine_scores) ** 2 + np.std(impostor_scores) ** 2)
    )

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": roc_auc,
        "d_prime": d_prime,
        "far": float(far_list[eer_idx]),
        "fvr": float(fvr_list[eer_idx]),
        "gar": 1.0 - float(fvr_list[eer_idx]),
        "gar_at_far_01": gar_at_far_01,
        "gar_at_far_001": gar_at_far_001,
        "genuine_mean": float(np.mean(genuine_scores)),
        "genuine_std": float(np.std(genuine_scores)),
        "impostor_mean": float(np.mean(impostor_scores)),
        "impostor_std": float(np.std(impostor_scores)),
        "num_genuine": len(genuine_scores),
        "num_impostor": len(impostor_scores),
    }


def run_4finger_test(
    dataset: str = "fvc2000",
    use_synthetic: bool = True,
    alpha: float = DEFAULT_CANCELABLE_ALPHA,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    intra_class_std: float = 0.02,
) -> Dict:
    """
    Run 4-finger biometric authentication test.

    Args:
        dataset: Dataset name ('fvc2000' or 'fvc2004')
        use_synthetic: Use synthetic embeddings if real data not available
        alpha: Cancelable transform alpha parameter
        threshold: Similarity threshold for authentication
        intra_class_std: Variation within same person (higher = harder)

    Returns:
        Dictionary with test results and metrics
    """
    print("=" * 70)
    print("4-FINGER BIOMETRIC AUTHENTICATION TEST")
    print("=" * 70)
    print(f"\nArchitecture:")
    print("  Raw Embedding (512-dim)")
    print("          |")
    print("          V")
    print("  Cancelable Transform R = alpha*R_bio + (1-alpha)*R_key")
    print("          |")
    print("          V")
    print("  Transformed Embedding")
    print("          |")
    print("  +-------+-------+")
    print("  |               |")
    print("  V               V")
    print(" PATH A:       PATH B:")
    print(" Cosine        PBKDF2 -> Key")
    print(" Similarity")
    print("          |")
    print("          V")
    print("  Authentication + Key")
    print(f"\nConfiguration:")
    print(f"  Alpha: {alpha}")
    print(f"  Threshold: {threshold}")
    print(f"  Intra-class std: {intra_class_std}")

    # Get dataset configuration
    config = get_dataset_config(dataset)
    data_dir = config["data_dir"]

    # Load data
    if use_synthetic:
        print(f"\nLoading synthetic embeddings for {dataset.upper()}...")
        data = generate_synthetic_embeddings(
            num_persons=config["num_persons"],
            embedding_dim=512,
            intra_class_std=intra_class_std,
        )
    else:
        print(f"\nLoading {dataset.upper()} data from {data_dir}...")
        data = load_fvc2000_4finger_data(
            data_dir=data_dir,
            enrollment_fingers=config["enrollment_fingers"],
            verification_fingers=config["verification_fingers"],
            num_persons=config["num_persons"],
        )

    enrollment_data = data["enrollment"]
    verification_data = data["verification"]

    num_persons = len(enrollment_data)
    print(f"  Persons: {num_persons}")

    # Initialize authenticator
    auth = BiometricAuthenticator(alpha=alpha, similarity_threshold=threshold)

    # Phase 1: Enrollment
    print(f"\n[Phase 1] Enrollment...")
    templates = {}
    for person_id in enrollment_data.keys():
        user_key = f"user_{person_id}"

        # Use first enrollment finger
        finger_id = list(enrollment_data[person_id].keys())[0]
        if use_synthetic:
            embedding = enrollment_data[person_id][finger_id]
        else:
            # Would load from file for real data
            continue

        template, params = auth.enroll(embedding, user_key)
        templates[person_id] = (template, params)

    print(f"  Enrolled: {len(templates)} templates")

    # Phase 2: Verification - Genuine matches
    print(f"\n[Phase 2] Genuine Verification (Path A: Cosine Similarity)...")
    genuine_scores = []
    key_derivation_success = 0
    for person_id in verification_data.keys():
        user_key = f"user_{person_id}"
        template, params = templates[person_id]

        # Use first verification finger
        finger_id = list(verification_data[person_id].keys())[0]
        if use_synthetic:
            probe_embedding = verification_data[person_id][finger_id]
        else:
            continue

        success, similarity, key = auth.verify(probe_embedding, template, params)
        genuine_scores.append(similarity)
        if success and key:
            key_derivation_success += 1

    print(f"  Genuine tests: {len(genuine_scores)}")
    print(f"  Mean score: {np.mean(genuine_scores):.4f}")
    print(
        f"  Key derivation success (Path B): {key_derivation_success}/{len(genuine_scores)}"
    )

    # Phase 3: Impostor matches
    print(f"\n[Phase 3] Impostor Verification...")
    impostor_scores = []
    impostor_scores = []
    person_ids = list(templates.keys())

    for _ in range(len(genuine_scores)):
        # Random different person
        person1, person2 = np.random.choice(person_ids, 2, replace=False)

        probe_embedding = verification_data[person2][
            list(verification_data[person2].keys())[0]
        ]
        template, params = templates[person1]

        success, similarity, key = auth.verify(probe_embedding, template, params)
        impostor_scores.append(similarity)

    print(f"  Impostor tests: {len(impostor_scores)}")
    print(f"  Mean score: {np.mean(impostor_scores):.4f}")

    # Compute metrics
    print(f"\n[Phase 4] Computing Metrics...")
    metrics = compute_metrics(genuine_scores, impostor_scores)

    # Print results
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nMetrics at EER threshold ({metrics['eer_threshold']:.4f}):")
    print(f"  EER (Equal Error Rate):     {metrics['eer'] * 100:.2f}%")
    print(f"  FAR (False Accept Rate):    {metrics['far'] * 100:.2f}%")
    print(f"  FVR (False Reject Rate):    {metrics['fvr'] * 100:.2f}%")
    print(f"  GAR (Genuine Accept Rate):  {metrics['gar'] * 100:.2f}%")

    print(f"\nAdditional Metrics:")
    print(f"  AUC:                        {metrics['auc']:.4f}")
    print(f"  d-prime:                   {metrics['d_prime']:.4f}")
    print(f"  GAR @ FAR=1%:               {metrics['gar_at_far_01'] * 100:.2f}%")
    print(f"  GAR @ FAR=0.1%:             {metrics['gar_at_far_001'] * 100:.2f}%")

    print(f"\nScore Statistics:")
    print(
        f"  Genuine - Mean: {metrics['genuine_mean']:.4f}, Std: {metrics['genuine_std']:.4f}"
    )
    print(
        f"  Impostor - Mean: {metrics['impostor_mean']:.4f}, Std: {metrics['impostor_std']:.4f}"
    )
    print(
        f"  Sample sizes - Genuine: {metrics['num_genuine']}, Impostor: {metrics['num_impostor']}"
    )

    return {
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "metrics": metrics,
    }


def plot_roc(results: Dict, save_path: str = "artifacts/plots/4finger_roc.png"):
    """Plot ROC curve for the results."""
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    genuine = np.array(results["genuine_scores"])
    impostor = np.array(results["impostor_scores"])

    all_scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])

    fpr, tpr, thresholds = roc_curve(labels, all_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Accept Rate (FAR)", fontsize=12)
    plt.ylabel("Genuine Accept Rate (GAR)", fontsize=12)
    plt.title("4-Finger Biometric Authentication ROC", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nROC curve saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="4-Finger Biometric Authentication Test"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fvc2000",
        choices=["fvc2000", "fvc2004"],
        help="Dataset to use (fvc2000 or fvc2004)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Use synthetic embeddings",
    )
    parser.add_argument("--real", action="store_true", help="Use real embeddings")
    parser.add_argument(
        "--alpha", type=float, default=0.6, help="Cancelable transform alpha"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2, help="Similarity threshold"
    )

    args = parser.parse_args()

    use_synthetic = not args.real
    results = run_4finger_test(
        dataset=args.dataset,
        use_synthetic=use_synthetic,
        alpha=args.alpha,
        threshold=args.threshold,
    )
    plot_roc(results)
