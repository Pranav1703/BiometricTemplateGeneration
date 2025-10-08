import os
import itertools
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Assume get_embeddings and other initializations are the same
from src.fingerprint.idv_inference import get_embeddings
from src.config import DATA_DIR

# ---------------------------
# Dataset path
# ---------------------------
DATASET_PATH = os.path.join(DATA_DIR, "CASIA-dataset")

# ---------------------------
# Load image paths (NO CHANGE)
# ---------------------------
def load_image_paths(dataset_path):
    image_dict = {}
    for id_folder in sorted(os.listdir(dataset_path)):
        id_path = os.path.join(dataset_path, id_folder)
        if not os.path.isdir(id_path):
            continue
        image_dict[id_folder] = {}
        for hand in ["L", "R"]:
            hand_path = os.path.join(id_path, hand)
            if os.path.exists(hand_path):
                imgs = [
                    os.path.join(hand_path, f)
                    for f in os.listdir(hand_path)
                    if f.endswith(".bmp")
                ]
                image_dict[id_folder][hand] = imgs
    return image_dict

# ---------------------------
# Compute embeddings (NO CHANGE)
# ---------------------------
def compute_embeddings(images, protected=False, user_key="default_key"):
    embeddings = {}
    total_images = sum(len(imgs) for hands in images.values() for imgs in hands.values())
    print(f"\nComputing {'protected' if protected else 'raw'} embeddings for {total_images} images...")

    for id_, hands in tqdm(images.items(), desc="IDs"):
        embeddings[id_] = {}
        for hand, img_list in hands.items():
            embeddings[id_][hand] = []
            for img_path in img_list:
                raw, prot = get_embeddings(img_path, user_key=user_key)
                embeddings[id_][hand].append(prot.numpy() if protected else raw.numpy())
    return embeddings


# ---------------------------
# Compute genuine and impostor scores (OPTIMIZED)
# ---------------------------
def compute_scores_optimized(embeddings):
    genuine_scores, impostor_scores = [], []
    ids = list(embeddings.keys())

    print("\nComputing genuine scores (optimized)...")
    for id_, hands in tqdm(embeddings.items(), desc="Genuine"):
        for hand, embs in hands.items():
            if len(embs) < 2:
                continue
            # Convert list of embeddings to a matrix
            embs_matrix = np.vstack(embs)
            # Compute all-vs-all similarity within the same identity
            sim_matrix = cosine_similarity(embs_matrix)
            # Get the upper triangle of the matrix, excluding the diagonal (k=1)
            # This gives us the scores for unique pairs (e1, e2), (e1, e3), etc.
            genuine_scores.extend(sim_matrix[np.triu_indices(len(embs), k=1)])

    print("\nComputing impostor scores (optimized)...")
    # Iterate through all unique pairs of different IDs
    for id1, id2 in tqdm(itertools.combinations(ids, 2), total=(len(ids)*(len(ids)-1))//2, desc="Impostor"):
        for hand in ["L", "R"]:
            embs1 = embeddings[id1].get(hand)
            embs2 = embeddings[id2].get(hand)
            
            if not embs1 or not embs2:
                continue

            # Convert lists of embeddings to matrices
            matrix1 = np.vstack(embs1)
            matrix2 = np.vstack(embs2)
            
            # THE KEY CHANGE: Compute all scores between the two matrices in one go!
            sim_matrix = cosine_similarity(matrix1, matrix2)
            
            # Flatten the resulting matrix of scores and add to our list
            impostor_scores.extend(sim_matrix.flatten())

    return np.array(genuine_scores), np.array(impostor_scores)


# ---------------------------
# Save scores (NO CHANGE)
# ---------------------------
def save_scores(prefix, genuine, impostor):
    output_dir = "artifacts/scores"
    os.makedirs(output_dir, exist_ok=True)
    genuine_path = os.path.join(output_dir, f"{prefix}_genuine.npy")
    impostor_path = os.path.join(output_dir, f"{prefix}_impostor.npy")
    np.save(genuine_path, genuine)
    np.save(impostor_path, impostor)
    print(f"{prefix.capitalize()} scores saved: {genuine_path}, {impostor_path}\n")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    images = load_image_paths(DATASET_PATH)
    
    # You can choose how many subjects to process for a quick test
    # For example, to test with the first 10 subjects:
    # ids_to_process = list(images.keys())[:10]
    # images = {id_: images[id_] for id_ in ids_to_process}

    # --- Raw Embeddings ---
    raw_embeddings = compute_embeddings(images, protected=False)
    raw_genuine, raw_impostor = compute_scores_optimized(raw_embeddings)
    save_scores("raw", raw_genuine, raw_impostor)

    # --- Protected Embeddings ---
    prot_embeddings = compute_embeddings(images, protected=True, user_key="a_secure_user_key")
    prot_genuine, prot_impostor = compute_scores_optimized(prot_embeddings)
    save_scores("protected", prot_genuine, prot_impostor)

    print("All computations completed successfully.")