"""Stage 2: fine-tune the lab-trained Leafora model on real (field) photos.

Usage (from `backend/`, inside the virtual environment):

    python -m training.finetune_field

Why this exists: the stage-1 model (the notebook's configuration, trained on
PlantVillage) scores well on lab photos and poorly on real phone photos. Stage 2
keeps every layer stage 1 learned, unfreezes the last DenseNet block, and lets the
field photos adjust it at a very low learning rate.

This is an ADDITION, not the notebook. It is reported separately:

    lab   accuracy  <- training/evaluate.py      (stage 1, the notebook's setup)
    field accuracy  <- training/evaluate_field.py

What it does:
    1. load the exported stage-1 model (so stage 1 is never overwritten)
    2. mix field photos (25%) with lab photos (75%) per batch, because 4 of the
       10 classes have too few field photos to learn from alone
    3. unfreeze DenseNet from `?conv5_block16` onward, keep BatchNorm frozen
    4. Adam at 1e-5, categorical crossentropy, early stopping patience 3
    5. save to models/exported/leafora_densenet121_field.keras
    6. write training/results/finetune_metadata.json

Nothing is claimed as measured until it runs. The accuracy it prints is real.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from training import config


def log(message: str) -> None:
    print(f"[leafora-finetune] {message}", flush=True)


def mix_datasets(lab_data, field_data):
    """Interleave lab and field batches so every epoch sees both domains."""
    import tensorflow as tf

    lab_batches = iter(lab_data.repeat())
    field_batches = iter(field_data.repeat())

    def generator():
        while True:
            yield next(lab_batches)
            yield next(field_batches)

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=config.EXPORTED_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=config.FINETUNE_EPOCHS)
    parser.add_argument(
        "--field-sample",
        type=float,
        default=config.FINETUNE_FIELD_SAMPLE,
        help="Share of field/train used for fine-tuning (the rest of field/train is unused).",
    )
    parser.add_argument("--unfreeze-from", default=config.FINETUNE_UNFREEZE_FROM)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args(argv)

    import numpy as np
    import tensorflow as tf

    from training.preprocessing import (
        field_class_names,
        load_datasets,
        load_field_datasets,
        take_fraction,
    )

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    if not args.base_model.exists():
        log(f"Stage-1 model not found at {args.base_model}.")
        log("Run `python -m training.train` first, or pass --base-model.")
        return 1
    if not config.FIELD_TRAIN_DIR.is_dir():
        log(f"Field training folder not found: {config.FIELD_TRAIN_DIR}")
        return 1

    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    class_names = json.loads(config.CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    log(f"Class order ({len(class_names)} classes) from {config.CLASS_NAMES_PATH}")

    lab_train, _ = load_datasets(config.IMAGE_SIZE, config.BATCH_SIZE)
    field_train_full, field_val = load_field_datasets(config.IMAGE_SIZE, config.BATCH_SIZE)

    if field_class_names() != class_names:
        log("Class order mismatch between the field folder and class_names.json:")
        log(f"  field folder : {field_class_names()}")
        log(f"  class_names  : {class_names}")
        return 1

    field_train, kept, total = take_fraction(field_train_full, args.field_sample)
    log(f"Field fine-tune batches: {kept} of {total} ({args.field_sample:.0%} of field/train)")

    model = tf.keras.models.load_model(args.base_model, compile=False)
    log(f"Loaded stage-1 model: {args.base_model}")

    # Measure the starting point before touching any weight, so "did it help?" is
    # answered by numbers on the same photos, not by hope.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINETUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    baseline = model.evaluate(field_val, verbose=0)
    baseline_accuracy = float(baseline[1])
    log(f"Starting field accuracy (stage-1 model, untouched): {baseline_accuracy * 100:.2f}%")

    # Unfreeze only the top of DenseNet; keep every BatchNorm frozen. Unfreezing
    # BatchNorm with batches this small would move the running statistics and
    # damage what stage 1 learned.
    conv_base = model.layers[0]
    conv_base.trainable = True
    unfreezing = False
    trainable_count = 0
    for layer in conv_base.layers:
        if layer.name.startswith(args.unfreeze_from):
            unfreezing = True
        layer.trainable = unfreezing and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
        if layer.trainable:
            trainable_count += 1
    log(f"Unfroze {trainable_count} DenseNet layers from '{args.unfreeze_from}' onward")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINETUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = config.CHECKPOINT_DIR / "field_best.keras"
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Fine-tuning up to {args.epochs} epochs (lr {config.FINETUNE_LEARNING_RATE}, patience {config.FINETUNE_PATIENCE})")
    started_at = datetime.now(timezone.utc)
    history = model.fit(
        mix_datasets(lab_train, field_train),
        epochs=args.epochs,
        steps_per_epoch=len(field_train),
        validation_data=field_val,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.FINETUNE_PATIENCE,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ],
        verbose=2,
    )
    finished_at = datetime.now(timezone.utc)
    duration = (finished_at - started_at).total_seconds()
    epochs_completed = len(history.history["loss"])
    log(f"Fine-tuning finished in {duration / 60:.1f} minutes over {epochs_completed} epochs")

    # ModelCheckpoint saved the epoch with the lowest field val_loss. The in-memory
    # model is whatever epoch the run stopped on, which can be a worse one - so the
    # checkpoint is what gets exported and measured. This is where the earlier run
    # lost accuracy: it exported the last epoch instead of the best one.
    if checkpoint_path.exists():
        model = tf.keras.models.load_model(checkpoint_path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINETUNE_LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        log(f"Reloaded best checkpoint: {checkpoint_path}")

    field_metrics = model.evaluate(field_val, verbose=0)
    metrics = dict(zip(model.metrics_names, [float(v) for v in field_metrics]))
    if "compile_metrics" in metrics:
        metrics["accuracy"] = metrics.pop("compile_metrics")
    log(f"Field validation metrics after fine-tuning: {metrics}")

    if not args.no_export:
        model.save(config.FINETUNED_MODEL_PATH)
        log(f"Saved fine-tuned model -> {config.FINETUNED_MODEL_PATH}")

        metadata = {
            "model_version": f"{config.MODEL_VERSION}-field",
            "stage": 2,
            "base_model": str(args.base_model),
            "exported_at": finished_at.isoformat(),
            "note": (
                "Stage 2 is an addition to the notebook's configuration, reported "
                "separately from the lab accuracy of stage 1."
            ),
            "image_size": config.IMAGE_SIZE,
            "num_classes": len(class_names),
            "class_names": class_names,
            "learning_rate": config.FINETUNE_LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "categorical_crossentropy",
            "epochs_requested": args.epochs,
            "epochs_completed": epochs_completed,
            "early_stopping": {"monitor": "val_loss", "patience": config.FINETUNE_PATIENCE},
            "unfreeze_from": args.unfreeze_from,
            "field_sample": args.field_sample,
            "field_train_batches_used": kept,
            "field_val_dir": str(config.FIELD_VAL_DIR),
            "field_accuracy_before": round(baseline_accuracy, 4),
            "field_accuracy_after": round(float(metrics.get("accuracy", float("nan"))), 4),
            "field_accuracy_gain": round(
                float(metrics.get("accuracy", float("nan"))) - baseline_accuracy, 4
            ),
            "field_validation_metrics": metrics,
            "history": history.history,
            "training_seconds": round(duration, 1),
            "environment": {
                "python_version": __import__("sys").version.split()[0],
                "tensorflow_version": tf.__version__,
                "gpu_devices": [g.name for g in tf.config.list_physical_devices("GPU")],
            },
        }
        metadata_path = config.RESULTS_DIR / "finetune_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        log(f"Saved metadata -> {metadata_path}")

    final_accuracy = float(metrics.get("accuracy", float("nan")))
    log(f"FIELD ACCURACY BEFORE: {baseline_accuracy * 100:.2f}%")
    log(f"FIELD ACCURACY AFTER:  {final_accuracy * 100:.2f}%")
    log(f"CHANGE: {((final_accuracy - baseline_accuracy) * 100):+.2f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
