import torch
import numpy as np
from models.fingerprint.preprocess_fingerprint import preprocess_fingerprint
from src.preprocess.iris import preprocess_image
from src.models.fingerprint.train import FingerprintEmbeddingNet
from src.models.Iris.train import IrisEmbeddingNet
from src.config import FINGERPRINT_MODEL_PATH, IRIS_MODEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load Models ======
finger_model = FingerprintEmbeddingNet()
finger_model.load_state_dict(torch.load(FINGERPRINT_MODEL_PATH, map_location=device))
finger_model.to(device).eval()

iris_model = IrisEmbeddingNet()
iris_model.load_state_dict(torch.load(IRIS_MODEL_PATH, map_location=device))
iris_model.to(device).eval()

# ====== Generate Template ======
def generate_template(finger_img, iris_img, method="concat", alpha=0.5):
    # Preprocess
    f_tensor = preprocess_fingerprint(finger_img).unsqueeze(0).to(device)
    i_tensor = preprocess_image(iris_img).unsqueeze(0).to(device)

    with torch.no_grad():
        f_emb = finger_model(f_tensor)   # [1, 256]
        i_emb = iris_model(i_tensor)     # [1, 256]

    if method == "concat":
        template = torch.cat((f_emb, i_emb), dim=1)  # [1, 512]
    elif method == "average":
        template = (f_emb + i_emb) / 2  # [1, 256]
    elif method == "weighted":
        template = alpha * f_emb + (1 - alpha) * i_emb
    else:
        raise ValueError("Method must be 'concat', 'average', or 'weighted'")

    return template.cpu()

# # ====== Example Usage ======
finger_img = "data/IRIS_and_FINGERPRINT_DATASET/1/fingerprint/1__M_Left_index_finger.BMP"
iris_img   = "data/IRIS_and_FINGERPRINT_DATASET/1/left/aeval1.bmp"

template = generate_template(finger_img, iris_img, method="concat")
print("\nMultimodal template shape:", template.shape)
# print("Multimodal Template (first 10 values):\n", template.flatten()[:10].numpy())
print("\n Multimodal Template:\n", template.numpy())

# Save template
np.save("user1_template.npy", template.numpy())


# dim test
# dummy = torch.randn(1, 3, 224, 224)  # RGB
# try:
#     finger_model(dummy)
#     print("Model expects 3 channels")
# except Exception as e:
#     print("Failed with 3 channels:", e)

# dummy = torch.randn(1, 1, 224, 224)  # Grayscale
# try:
#     finger_model(dummy)
#     print("Model expects 1 channel")
# except Exception as e:
#     print("Failed with 1 channel:", e)
