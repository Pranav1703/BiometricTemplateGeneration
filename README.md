# Biometric template generation

A multimodal biometric recognition system that uses both fingerprint and iris images to identify individuals. Separate deep learning models extract embeddings for each modality, which are then fused to form a compact biometric template — a numerical representation that can be stored securely and used for efficient, privacy-preserving authentication or identification.

### Venvs Creation

- for cpu

```bash
    conda env create -f envs\environment-cpu.yml
```

- for gpu

```bash
    conda env create -f envs\environment-gpu.yml
```

### Venv activation

```bash
    conda activate biometric-env
```

### Dataset prep
Dataset - https://www.kaggle.com/datasets/ninadmehendale/multimodal-iris-fingerprint-biometric-data

run this command to generate coresponding csv files for fingerprint and iris images.

```bash
python data/gen-labels.py --root ./IRIS_and_FINGERPRINT_DATASET --outdir ./labels
```
## Project Structure
```
BiometricTemplateGeneration/
├── data/                          # Dataset directory
│   ├── labels/                    # Generated labels for training/validation
│   │   ├── fingerprint_train.csv
│   │   ├── fingerprint_val.csv
│   │   ├── iris_train.csv
│   │   └── iris_val.csv
│   └── IRIS and FINGERPRINT DATASET/  # Raw biometric dataset
│       ├── 1/                     # Subject 1
│       │   ├── Fingerprint/       # Fingerprint images (.BMP)
│       │   │   ├── 1__M_Right_thumb_finger.BMP
│       │   │   ├── 1__M_Right_index_finger.BMP
│       │   │   ├── 1__M_Right_middle_finger.BMP
│       │   │   ├── 1__M_Right_ring_finger.BMP
│       │   │   ├── 1__M_Right_little_finger.BMP
│       │   │   ├── 1__M_Left_thumb_finger.BMP
│       │   │   ├── 1__M_Left_index_finger.BMP
│       │   │   ├── 1__M_Left_middle_finger.BMP
│       │   │   ├── 1__M_Left_ring_finger.BMP
│       │   │   └── 1__M_Left_little_finger.BMP
│       │   ├── right/             # Right eye iris images (.bmp)
│       │   │   ├── aevar1.bmp
│       │   │   ├── aevar2.bmp
│       │   │   ├── aevar3.bmp
│       │   │   ├── aevar4.bmp
│       │   │   └── aevar5.bmp
│       │   └── left/              # Left eye iris images (.bmp)
│       ├── 2/                     # Subject 2
│       ├── 3/                     # Subject 3
│       ├── ...                    # Subjects 4-45
│       └── 45/                    # Subject 45
├── artifacts/                     # Generated artifacts and outputs
│   ├── plots/                     # Training plots and visualizations
│   │   └── tensorboard/           # TensorBoard logs
│   │       └── 20250819-031405/
│   ├── models/                    # Trained model files
│   │   └── fingerprint_embedding_model.pth
│   └── logs/                      # Training logs
│       └── training/
│           ├── 20250819_031353/
│           └── 20250819_031259/
├── src/                           # Source code
│   ├── config.py                  # Configuration settings
│   ├── init.py                    # Package initialization
│   ├── models/                    # Model implementations
│   │   ├── init.py
│   │   ├── fingerprint/           # Fingerprint model components
│   │   │   ├── train.py           # Fingerprint training script
│   │   │   ├── test.py            # Fingerprint testing script
│   │   │   └── eval_threshold.py  # Threshold evaluation
│   │   ├── Iris/                  # Iris model components
│   │   │   ├── train.py           # Iris training script
│   │   │   ├── test.py            # Iris testing script
│   │   │   └── eval_threshold.py  # Threshold evaluation
│   ├── preprocess/                # Data preprocessing modules
│   │   ├── init.py
│   │   ├── fingerprint.py         # Fingerprint preprocessing
│   │   └── iris.py                # Iris preprocessing
│   ├── utils/                     # Utility functions
│   │   ├── init.py
│   │   ├── Dataset_Loader.py      # Dataset loading utilities
│   │   ├── gen_labels.py          # Label generation script
│   │   ├── logger.py              # Logging utilities
│   │   └── plot.py                # Plotting utilities
├── envs/                          # Environment configurations
│   ├── environment-cpu.yml        # CPU environment setup
│   └── environment-gpu.yml        # GPU environment setup
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── .gitignore                     # Git ignore rules
```
