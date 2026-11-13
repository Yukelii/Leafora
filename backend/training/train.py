"""Reproduce `tomato-leaf-disease-94-accuracy.ipynb` and export the model.

Usage (from the `backend/` folder, inside the virtual environment):

    python -m training.train --epochs 100

What it does, in the notebook's order:
    1. discover class names from the dataset (never hard-coded)
    2. load train/val with image_dataset_from_directory, 256x256, batch 32,
       labels inferred, categorical, pixels divided by 255.0
    3. build frozen DenseNet121 (ImageNet, include_top=False, pooling='avg')
       + BatchNormalization -> Dense(256, relu) -> Dropout(0.35)
       -> BatchNormalization -> Dense(120, relu) -> Dense(num_classes, softmax)
    4. compile Adam(1e-4) / categorical_crossentropy / accuracy
    5. fit with EarlyStopping(patience=0, monitor='val_loss'), max 100 epochs
    6. evaluate on the validation set
    7. export the model, class_names.json, metadata and training history

Honesty note: the notebook's title claims 94% accuracy. This script measures the
real number on this machine and this dataset copy. Do not copy the claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from training import config
from training.preprocessing import (
    count_images,
    discover_class_names,
    load_datasets,
    save_class_names,
)


def log(message: str) -> None:
    print(f"[leafora-train] {message}", flush=True)


def environment_report() -> dict[str, object]:
    """Record the exact environment, since training happens on local hardware."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    info: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "bundled"),
        "gpu_devices": [g.name for g in gpus],
        "gpu_available": bool(gpus),
        "cuda_built": bool(tf.test.is_built_with_cuda()),
        "random_seed": config.RANDOM_SEED,
    }
    if not gpus:
        info["gpu_note"] = (
            "TensorFlow did not detect a GPU. On Windows this usually means the "
            "TensorFlow build is CPU-only or CUDA/cuDNN for TF 2.19 is not "
            "installed. Training will run on CPU."
        )
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs", type=int, default=config.MAX_EPOCHS,
        help=f"Maximum epochs (notebook used {config.MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Train and evaluate but do not write the exported model.",
    )
    args = parser.parse_args(argv)

    import numpy as np
    import tensorflow as tf

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.ml.model import build_model

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = environment_report()
    log(f"TensorFlow {env['tensorflow_version']} on Python {env['python_version']}")
    log(f"GPU devices: {env['gpu_devices'] or 'none (CPU training)'}")

    class_names = discover_class_names(config.TRAIN_DIR)
    num_classes = len(class_names)
    log(f"Dataset: {config.TRAIN_DIR}")
    log(f"Classes ({num_classes}): {class_names}")
    log(f"Train images: {count_images(config.TRAIN_DIR)}")
    log(f"Val images:   {count_images(config.VAL_DIR)}")

    if config.VAL_DIR.is_dir():
        val_classes = discover_class_names(config.VAL_DIR)
        if val_classes != class_names:
            log(f"WARNING: validation folder classes differ: {val_classes}")

    train_data, val_data = load_datasets(config.IMAGE_SIZE, config.BATCH_SIZE)

    model = build_model(num_classes=num_classes, image_size=config.IMAGE_SIZE)
    model.summary(print_fn=lambda line: log(line))

    checkpoint_path = config.CHECKPOINT_DIR / "best.keras"
    callbacks = [
        # Notebook: EarlyStopping(patience=0) -> no restore_best_weights.
        tf.keras.callbacks.EarlyStopping(
            monitor=config.MONITOR,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=False,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=config.MONITOR,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(config.RESULTS_DIR / "training_log.csv")),
    ]

    log(f"Training up to {args.epochs} epochs (batch {config.BATCH_SIZE}, lr {config.LEARNING_RATE})")
    started = time.perf_counter()
    history = model.fit(
        train_data,
        epochs=args.epochs,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=2,
    )
    duration = time.perf_counter() - started
    log(f"Training finished in {duration / 60:.1f} minutes over {len(history.history['loss'])} epochs")

    log("Evaluating on the validation set")
    evaluation = model.evaluate(val_data, verbose=0)
    # Keras reports the metric name as 'compile_metrics'; label it 'accuracy'.
    metrics = dict(zip(model.metrics_names, [float(v) for v in evaluation]))
    if "compile_metrics" in metrics:
        metrics["accuracy"] = metrics.pop("compile_metrics")
    log(f"Validation metrics: {metrics}")

    history_path = config.RESULTS_DIR / "history.json"
    history_path.write_text(json.dumps(history.history, indent=2) + "\n", encoding="utf-8")

    if not args.no_export:
        # Save the class mapping first: the API validates model/class agreement.
        save_class_names(class_names, config.CLASS_NAMES_PATH)
        model.save(config.EXPORTED_MODEL_PATH)
        log(f"Saved model -> {config.EXPORTED_MODEL_PATH}")
        log(f"Saved class names -> {config.CLASS_NAMES_PATH}")

        metadata = {
            "model_version": config.MODEL_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "architecture": "DenseNet121 (ImageNet, frozen) + custom head",
            "image_size": config.IMAGE_SIZE,
            "num_classes": num_classes,
            "class_names": class_names,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
            "max_epochs": args.epochs,
            "epochs_completed": len(history.history["loss"]),
            "early_stopping": {
                "monitor": config.MONITOR,
                "patience": config.EARLY_STOPPING_PATIENCE,
                "restore_best_weights": False,
            },
            "normalization": "pixel / 255.0",
            "dataset_dir": str(config.DATASET_DIR),
            "train_images": count_images(config.TRAIN_DIR),
            "val_images": count_images(config.VAL_DIR),
            "validation_metrics": metrics,
            "training_seconds": round(duration, 1),
            "environment": env,
        }
        metadata_path = config.RESULTS_DIR / "model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        log(f"Saved metadata -> {metadata_path}")

    log("Done. Measured validation accuracy is in results/model_metadata.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
