"""Measure the decision layer on real field photos.

Two questions this answers with numbers, not opinions:

1. Of the wrong confident answers the model gives, how many does the confidence
   gate convert into "uncertain" (i.e. stop telling the user a wrong disease)?
2. How often is the symptom checklist alone right, when answers are supplied?

The checklist cannot be measured honestly without human answers, so this script
measures it with a *simulated* answerer that answers from the true label. Those
numbers are reported separately and labelled as an upper bound, never as
accuracy of the system.

Usage:
    python -m training.evaluate_decision_layer
    python -m training.evaluate_decision_layer --limit 400
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.decision.diagnosis import DiagnosisEngine, DiagnosisSettings
from app.decision.symptoms import ALL_CLASSES, QUESTIONS
from training.config import RESULTS_DIR, FIELD_VAL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str(settings.model_path),
        help="Exported model to evaluate (default: the one the API loads).",
    )
    parser.add_argument("--val-dir", default=str(FIELD_VAL_DIR))
    parser.add_argument("--limit", type=int, default=0, help="0 means every image.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tag", default="decision_layer")
    return parser.parse_args()


def list_images(val_dir: Path) -> tuple[list[Path], list[str], list[str]]:
    paths: list[Path] = []
    labels: list[str] = []
    class_names = sorted(entry.name for entry in val_dir.iterdir() if entry.is_dir())
    for class_name in class_names:
        for file_path in sorted((val_dir / class_name).glob("*")):
            if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                paths.append(file_path)
                labels.append(class_name)
    return paths, labels, class_names


def main() -> int:
    args = parse_args()
    val_dir = Path(args.val_dir)
    if not val_dir.is_dir():
        print(f"[leafora-decision] Field validation folder not found: {val_dir}")
        return 1

    import tensorflow as tf

    model_path = Path(args.model)
    print(f"[leafora-decision] Model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    class_names = list(ALL_CLASSES)

    paths, labels, found = list_images(val_dir)
    if found and sorted(found) != sorted(class_names):
        print(
            "[leafora-decision] WARNING: folder classes differ from the trained "
            f"classes.\nfolder={found}\ntrained={class_names}"
        )
    if args.limit:
        paths, labels = paths[: args.limit], labels[: args.limit]
    print(f"[leafora-decision] Images: {len(paths)}")

    def load(path: str, label: str) -> tuple[np.ndarray, str]:
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, (256, 256))
        return tf.cast(image, tf.float32) / 255.0, label

    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(p) for p in paths], labels)
    ).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    probabilities = model.predict(dataset, verbose=0)
    index_of = {name: index for index, name in enumerate(class_names)}

    gate = DiagnosisEngine(DiagnosisSettings())
    total = len(paths)
    correct = 0
    wrong_confident = 0
    wrong_turned_uncertain = 0
    right_confident = 0
    wrong_still_confident = 0
    per_class_wrong: dict[str, int] = {}

    for probs, true_label in zip(probabilities, labels):
        photo = {name: float(probs[index_of[name]]) for name in class_names}
        verdict = gate.evaluate(photo)
        raw_top = max(photo, key=photo.get)
        if raw_top == true_label:
            correct += 1
        else:
            per_class_wrong[true_label] = per_class_wrong.get(true_label, 0) + 1
        if verdict.status == "confident":
            if verdict.predicted_class == true_label:
                right_confident += 1
            else:
                wrong_still_confident += 1
        elif raw_top != true_label:
            wrong_turned_uncertain += 1

    wrong_total = total - correct
    raw_accuracy = correct / total if total else 0.0
    answered = right_confident + wrong_still_confident
    answered_accuracy = right_confident / answered if answered else 0.0
    coverage = answered / total if total else 0.0

    summary: dict[str, Any] = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "val_dir": str(val_dir),
        "images": total,
        "raw_model_accuracy": round(raw_accuracy, 6),
        "with_gate": {
            "answered": answered,
            "coverage": round(coverage, 6),
            "answered_accuracy": round(answered_accuracy, 6),
            "right_confident": right_confident,
            "wrong_confident": wrong_still_confident,
            "wrong_turned_uncertain": wrong_turned_uncertain,
        },
        "gate_settings": {
            "min_confidence": gate.settings.min_confidence,
            "min_margin": gate.settings.min_margin,
        },
        "note": (
            "Measured on the field validation split. Early stopping monitored this "
            "same split during fine-tuning, so these are validation numbers, not a "
            "fully untouched test result."
        ),
    }

    lines = [
        "=" * 62,
        "LEAFORA - DECISION LAYER ON FIELD VALIDATION IMAGES",
        "=" * 62,
        f"Images                       : {total}",
        f"Raw model accuracy           : {raw_accuracy * 100:.2f}%",
        f"Wrong answers by the model   : {wrong_total}",
        "",
        f"After the confidence gate ({gate.settings.min_confidence:.2f} / "
        f"margin {gate.settings.min_margin:.2f}):",
        f"  Answers given              : {answered} ({coverage * 100:.1f}% of photos)",
        f"  Accuracy of those answers  : {answered_accuracy * 100:.2f}%",
        f"  Wrong and still answered   : {wrong_still_confident}",
        f"  Wrong, now 'uncertain'     : {wrong_turned_uncertain}",
        "=" * 62,
    ]
    print("\n".join(lines))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.tag}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[leafora-decision] Wrote {out_path}")

    # The checklist alone, simulated from the true label. Upper bound only.
    checklist = simulate_checklist_upper_bound(paths, labels)
    checklist_path = RESULTS_DIR / f"{args.tag}_checklist_upper_bound.json"
    checklist_path.write_text(json.dumps(checklist, indent=2), encoding="utf-8")
    print(
        "[leafora-decision] Checklist upper bound (simulated answers): "
        f"{checklist['accuracy'] * 100:.2f}% over {checklist['images']} images"
    )
    print(f"[leafora-decision] Wrote {checklist_path}")
    return 0


def best_option_for(class_name: str) -> dict[str, str]:
    """Pick the option each question would give if the true disease answered.

    This is why the result is an upper bound: a perfect answerer, not a farmer.
    """
    from app.decision.symptoms import score_answers

    answers: dict[str, str] = {}
    for question in QUESTIONS:
        best_key = ""
        best_score = -1.0
        for option in question.options:
            scores = score_answers({question.key: option.key})
            value = scores.get(class_name, 0.0)
            if value > best_score:
                best_score = value
                best_key = option.key
        if best_key:
            answers[question.key] = best_key
    return answers


def simulate_checklist_upper_bound(paths: list[Path], labels: list[str]) -> dict[str, Any]:
    from app.decision.symptoms import score_answers

    cache: dict[str, dict[str, str]] = {}
    correct = 0
    for label in labels:
        answers = cache.get(label)
        if answers is None:
            answers = best_option_for(label)
            cache[label] = answers
        scores = score_answers(answers)
        if max(scores, key=scores.get) == label:
            correct += 1
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "images": len(labels),
        "accuracy": round(correct / len(labels), 6) if labels else 0.0,
        "warning": (
            "UPPER BOUND ONLY. Answers were simulated from the true label, so this "
            "measures whether the checklist is internally consistent, not whether a "
            "real user would be diagnosed correctly. Do not report as system accuracy."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
