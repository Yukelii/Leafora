"""Measure a Leafora model on real (field) photos — the lab-vs-field gap.

Usage (from `backend/`, inside the virtual environment):

    python -m training.evaluate_field                      # the lab-trained model
    python -m training.evaluate_field --model models/exported/leafora_densenet121_field.keras --tag field_model

Why this exists: the lab-trained model scores ~93% on PlantVillage-style photos.
PlantVillage leaves are flat, evenly lit and photographed on a plain background;
a real phone photo is not. This script measures that difference instead of hiding
it, and writes the number to `training/results/`.

It never trains and never changes the model. It only reports.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from training import config


def log(message: str) -> None:
    print(f"[leafora-field] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=config.EXPORTED_MODEL_PATH)
    parser.add_argument("--class-names", type=Path, default=config.CLASS_NAMES_PATH)
    parser.add_argument(
        "--tag",
        default="lab_model",
        help="Used in the output file name: field_evaluation_<tag>.json",
    )
    args = parser.parse_args(argv)

    import numpy as np
    import tensorflow as tf

    from training.preprocessing import field_class_names, load_field_datasets

    if not args.model.exists():
        log(f"Model not found at {args.model}. Train one first.")
        return 1
    if not config.FIELD_VAL_DIR.is_dir():
        log(f"Field validation folder not found: {config.FIELD_VAL_DIR}")
        return 1

    class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(args.model, compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    _, field_val = load_field_datasets(config.IMAGE_SIZE, config.BATCH_SIZE)

    # Guard against silently mislabelled predictions: the folder order must match
    # the class order the model was trained with.
    dataset_classes = field_class_names()
    if dataset_classes != class_names:
        log("Class order mismatch between the field folder and class_names.json:")
        log(f"  field folder : {dataset_classes}")
        log(f"  class_names  : {class_names}")
        return 1

    log(f"Model: {args.model}")
    log(f"Field validation images: {sum(1 for _ in field_val.unbatch())}")

    evaluation = model.evaluate(field_val, verbose=0)
    metrics = dict(zip(model.metrics_names, [float(v) for v in evaluation]))
    if "compile_metrics" in metrics:
        metrics["accuracy"] = metrics.pop("compile_metrics")

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    for images, labels in field_val:
        probabilities = model.predict(images, verbose=0)
        y_pred.extend(int(i) for i in np.argmax(probabilities, axis=1))
        y_true.extend(int(i) for i in np.argmax(np.asarray(labels), axis=1))
        confidences.extend(float(v) for v in np.max(probabilities, axis=1))

    from sklearn.metrics import classification_report, confusion_matrix

    labels_present = sorted(set(y_true) | set(y_pred))
    names_present = [class_names[i] for i in labels_present]
    matrix = confusion_matrix(y_true, y_pred, labels=labels_present)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels_present,
        target_names=names_present,
        digits=4,
        zero_division=0,
    )

    per_class_accuracy = {}
    for index in labels_present:
        support = sum(1 for value in y_true if value == index)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == index and p == index)
        per_class_accuracy[class_names[index]] = {
            "support": support,
            "correct": correct,
            "accuracy": round(correct / support, 4) if support else None,
        }

    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"field_classification_report_{args.tag}.txt"
    report_path.write_text(report_text, encoding="utf-8")

    header = "true\\pred," + ",".join(names_present)
    rows = [header] + [
        names_present[i] + "," + ",".join(str(int(v)) for v in row)
        for i, row in enumerate(matrix)
    ]
    (results_dir / f"field_confusion_matrix_{args.tag}.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

    summary = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "field (real photos)",
        "field_val_dir": str(config.FIELD_VAL_DIR),
        "model_path": str(args.model),
        "model_version": config.MODEL_VERSION,
        "num_classes": len(class_names),
        "class_names": class_names,
        "field_validation_images": len(y_true),
        "metrics": metrics,
        "per_class": per_class_accuracy,
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else None,
        "confusion_matrix": matrix.tolist(),
        "environment": {
            "python_version": __import__("sys").version.split()[0],
            "tensorflow_version": tf.__version__,
            "gpu_devices": [g.name for g in tf.config.list_physical_devices("GPU")],
        },
    }
    summary_path = results_dir / f"field_evaluation_{args.tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    log(report_text)
    log(f"FIELD ACCURACY: {metrics.get('accuracy', float('nan')) * 100:.2f}%")
    log(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
