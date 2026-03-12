import os
import itertools
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# --- SPECIFIC IMPORTS ---
# We alias them so we can switch between them easily
from src.fingerprint.idv_inference_CASIA import get_embeddings as get_embeddings_casia

from src.fingerprint.idv_inference_FVC2004 import get_embeddings as get_embeddings_fvc

from src.config import DATA_DIR, SCORES_DIR, FVC2004_DB1A_DIR

# ---------------------------
# Dataset Configurations
# ---------------------------
DATASET_CONFIGS = {
    "casia": {
        "path": os.path.join(DATA_DIR, "CASIA-dataset"),
        "inference_fn": get_embeddings_casia
    },
    "fvc2004": {
        "path": FVC2004_DB1A_DIR,
        "inference_fn": get_embeddings_fvc
    },
}

# ---------------------------
# Load image paths
# ---------------------------
def load_image_paths(dataset_path, dataset_type):
    image_dict = {}
    if dataset_type == "casia":
        for id_folder in sorted(os.listdir(dataset_path)):
            id_path = os.path.join(dataset_path, id_folder)
            if not os.path.isdir(id_path): continue
            for hand in ["L", "R"]:
                hand_path = os.path.join(id_path, hand)
                if os.path.exists(hand_path):
                    unique_id = f"{id_folder}_{hand}"
                    imgs = [os.path.join(hand_path, f) for f in os.listdir(hand_path) if f.endswith(".bmp")]
                    if imgs: image_dict[unique_id] = imgs
    elif dataset_type == "fvc2004":
        for f in sorted(os.listdir(dataset_path)):
            if f.endswith((".bmp", ".tif", ".png")):
                identity = f.split('_')[0]
                if identity not in image_dict: image_dict[identity] = []
                image_dict[identity].append(os.path.join(dataset_path, f))
    return image_dict

# ---------------------------
# Compute embeddings (Updated to use inference_fn)
# ---------------------------
def compute_embeddings(image_dict, inference_fn, protected=False, user_key="default_key"):
    embeddings = {}
    total_images = sum(len(imgs) for imgs in image_dict.values())
    mode = 'protected' if protected else 'raw'
    print(f"\nExtracting {mode} features using {inference_fn.__module__}...")

    for id_, img_list in tqdm(image_dict.items(), desc=f"Processing {mode}"):
        embeddings[id_] = []
        for img_path in img_list:
            # Call the specific inference function passed as an argument
            raw, prot = inference_fn(img_path, user_key=user_key)
            emb = prot.numpy() if protected else raw.numpy()
            embeddings[id_].append(emb.flatten())
    return embeddings

# ---------------------------
# Scoring Logic (No Change)
# ---------------------------
def compute_scores_optimized(embeddings):
    gen_scores, imp_scores = [], []
    ids = list(embeddings.keys())

    for id_ in tqdm(ids, desc="Genuine Pairs"):
        embs = embeddings[id_]
        if len(embs) < 2: continue
        sim_matrix = cosine_similarity(np.vstack(embs))
        gen_scores.extend(sim_matrix[np.triu_indices(len(embs), k=1)])

    for id1, id2 in tqdm(itertools.combinations(ids, 2), total=(len(ids)*(len(ids)-1))//2, desc="Impostor Pairs"):
        sim_matrix = cosine_similarity(np.vstack(embeddings[id1]), np.vstack(embeddings[id2]))
        imp_scores.extend(sim_matrix.flatten())

    return np.array(gen_scores), np.array(imp_scores)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Dataset Fingerprint Evaluation")
    parser.add_argument("--dataset", type=str, required=True, choices=["casia", "fvc2004"])
    args = parser.parse_args()

    # Get configuration based on dataset choice
    config = DATASET_CONFIGS[args.dataset]
    dataset_path = config["path"]
    inference_function = config["inference_fn"]

    # 1. Load data structure
    img_dict = load_image_paths(dataset_path, args.dataset)

    # 2. Extract and Score Raw
    raw_embs = compute_embeddings(img_dict, inference_function, protected=False)
    g_raw, i_raw = compute_scores_optimized(raw_embs)
    
    # 3. Extract and Score Protected
    prot_embs = compute_embeddings(img_dict, inference_function, protected=True, user_key="test_key")
    g_prot, i_prot = compute_scores_optimized(prot_embs)

    # Save results
    os.makedirs(SCORES_DIR, exist_ok=True)
    np.save(os.path.join(SCORES_DIR, f"{args.dataset}_raw_gen.npy"), g_raw)
    np.save(os.path.join(SCORES_DIR, f"{args.dataset}_raw_imp.npy"), i_raw)
    np.save(os.path.join(SCORES_DIR, f"{args.dataset}_prot_gen.npy"), g_prot)
    np.save(os.path.join(SCORES_DIR, f"{args.dataset}_prot_imp.npy"), i_prot)

    print(f"\nDone! Scores saved in {SCORES_DIR}")