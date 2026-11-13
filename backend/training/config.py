"""Training configuration. Mirrors `tomato-leaf-disease-94-accuracy.ipynb`.

Every value below is traceable to the notebook; where the notebook hard-coded
something we now derive it, and we say so.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repo root (this file lives in backend/training/).
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# --- dataset ---------------------------------------------------------------
#: Notebook used `/kaggle/input/tomatoleaf/tomato/train` and `.../val`.
#: This project's copy is `<repo>/tomato/train` and `<repo>/tomato/val`.
DATASET_DIR = Path(os.getenv("DATASET_DIR", REPO_ROOT / "tomato"))
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

# --- training hyperparameters (from the notebook) --------------------------
IMAGE_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 100  # notebook: model.fit(..., epochs=100)
#: Notebook: EarlyStopping(patience=0) -> stop on the first epoch without
#: improvement in validation loss, and the notebook does NOT set
#: restore_best_weights, so the final weights are the last epoch's.
EARLY_STOPPING_PATIENCE = 0
MONITOR = "val_loss"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "1337"))  # numpy seed used by cnn_train.py

# --- outputs ---------------------------------------------------------------
CHECKPOINT_DIR = BACKEND_ROOT / "training" / "checkpoints"
EXPORT_DIR = BACKEND_ROOT / "models" / "exported"
RESULTS_DIR = BACKEND_ROOT / "training" / "results"

#: Name of the exported Keras model from stage 1 (lab data).
#: `MODEL_PATH` in the API points here until stage 2 is measured.
EXPORTED_MODEL_NAME = "leafora_densenet121.keras"
EXPORTED_MODEL_PATH = EXPORT_DIR / EXPORTED_MODEL_NAME

#: Where the API reads the trained class order from.
CLASS_NAMES_PATH = BACKEND_ROOT / "app" / "ml" / "class_names.json"

# --- field data (real photos) ---------------------------------------------
#: Lab-vs-field gap work. The lab model is measured on this set, then fine-tuned on
#: its train split and re-measured on its val split.
#: Layout matches the lab set: <dir>/train/<class>/ and <dir>/val/<class>/.
FIELD_DATASET_DIR = Path(os.getenv("FIELD_DATASET_DIR", REPO_ROOT / "datasets" / "tomato_field"))
FIELD_TRAIN_DIR = FIELD_DATASET_DIR / "train"
FIELD_VAL_DIR = FIELD_DATASET_DIR / "val"

# --- fine-tuning (stage 2, NOT part of the notebook) -----------------------
#: Stage 2 is an addition, not the notebook's configuration. The base stage stays
#: byte-for-byte the notebook's: frozen DenseNet121, Adam 1e-4.
#: Stage 2 unfreezes the top of DenseNet and learns slowly so the field photos
#: adjust the features without erasing what the lab images taught.
FINETUNE_LEARNING_RATE = float(os.getenv("FINETUNE_LEARNING_RATE", "1e-5"))
#: 15 epochs was not enough: the measured run improved val_loss on every single
#: epoch right up to the cap, so it was cut off mid-learning. 25 gives it room to
#: converge while still fitting in a free Colab session; early stopping ends it
#: sooner when field val_loss stops improving.
FINETUNE_EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "25"))
#: Here patience > 0 on purpose: the field set is small, so a couple of noisy
#: epochs must not end the run. This differs from the notebook and is documented
#: as a deviation.
FINETUNE_PATIENCE = int(os.getenv("FINETUNE_PATIENCE", "8"))
#: Last DenseNet block to unfreeze (everything from here on becomes trainable).
#: conv5_block1 unfreezes the whole last block, not just its tail, because the
#: tail alone could not close the lab-to-field gap.
FINETUNE_UNFREEZE_FROM = os.getenv("FINETUNE_UNFREEZE_FROM", "conv5_block1")
#: Share of field/train used for fine-tuning. 1.0 = use all of it.
#: field/val is a separate folder and stays unseen either way, so using all of
#: field/train is still an honest measurement - and 25% was throwing away most of
#: the real photos we have.
FINETUNE_FIELD_SAMPLE = float(os.getenv("FINETUNE_FIELD_SAMPLE", "1.0"))

#: The stage-2 model is exported under its own name so the lab model is never lost.
FINETUNED_MODEL_NAME = "leafora_densenet121_field.keras"
FINETUNED_MODEL_PATH = EXPORT_DIR / FINETUNED_MODEL_NAME

MODEL_VERSION = os.getenv("MODEL_VERSION", "leafora-densenet121-v1")
