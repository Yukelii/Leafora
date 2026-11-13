"""Evaluate an exported Leafora model and write a full report.

Usage (from `backend/`):

    python -m training.evaluate

Produces in `training/results/`:
    evaluation.json      overall metrics + environment
    classification_report.txt
    confusion_matrix.csv
    model_metadata.json  (if the model was exported)

Reports what is actually measured. Nothing here is a copied claim.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from training import config


def log(message: str) -> None:
    print(f"[leafora-eval] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=config.EXPORTED_MODEL_PATH)
    parser.add_argument("--class-names", type=Path, default=config.CLASS_NAMES_PATH)
    args = parser.parse_args(argv)

    import numpy as np
    import tensorflow as tf

    from training.preprocessing import load_datasets

    if not args.model.exists():
        log(f"Model not found at {args.model}. Run `python -m training.train` first.")
        return 1

    class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
    log(f"Model: {args.model}")
    log(f"Classes ({len(class_names)}): {class_names}")

    model = tf.keras.models.load_model(args.model, compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    _, val_data = load_datasets(config.IMAGE_SIZE, config.BATCH_SIZE)

    started = datetime.now(timezone.utc)
    evaluation = model.evaluate(val_data, verbose=0)
    metrics = dict(zip(model.metrics_names, [float(v) for v in evaluation]))
    if "compile_metrics" in metrics:
        metrics["accuracy"] = metrics.pop("compile_metrics")
    log(f"Validation metrics: {metrics}")

    # Collect predictions for the confusion matrix and per-class scores.
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in val_data:
        probabilities = model.predict(images, verbose=0)
        y_pred.extend(int(i) for i in np.argmax(probabilities, axis=1))
        y_true.extend(int(i) for i in np.argmax(np.asarray(labels), axis=1))

    labels_present = sorted(set(y_true) | set(y_pred))
    names_present = [class_names[i] for i in labels_present]

    from sklearn.metrics import classification_report, confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=labels_present)
    report_text = classification_report(
        y_true, y_pred, labels=labels_present, target_names=names_present, digits=4, zero_division=0
    )

    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    header = "true\\pred," + ",".join(names_present)
    rows = [header] + [
        names_present[i] + "," + ",".join(str(int(v)) for v in row)
        for i, row in enumerate(matrix)
    ]
    (results_dir / "confusion_matrix.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    summary = {
        "evaluated_at": started.isoformat(),
        "model_path": str(args.model),
        "model_version": config.MODEL_VERSION,
        "num_classes": len(class_names),
        "class_names": class_names,
        "validation_samples": len(y_true),
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "environment": {
            "python_version": __import__("sys").version.split()[0],
            "tensorflow_version": tf.__version__,
            "gpu_devices": [g.name for g in tf.config.list_physical_devices("GPU")],
        },
    }
    (results_dir / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    log(report_text)
    log(f"Wrote {results_dir / 'evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
