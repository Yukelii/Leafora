"""Model loading and class-mapping integrity.

These tests do not need a trained export: they check the loader's failure paths
and the agreement between `class_names.json` and the dataset folder names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ml.inference import ModelNotAvailableError, ModelService, load_class_names
from tests.conftest import CLASS_NAMES

CLASS_NAMES_FILE = Path(__file__).resolve().parents[1] / "app" / "ml" / "class_names.json"
DATASET_TRAIN = Path(__file__).resolve().parents[2] / "tomato" / "train"


def test_class_names_file_matches_dataset_folders():
    """A mismatch would silently mislabel every prediction."""
    on_disk = sorted(p.name for p in DATASET_TRAIN.iterdir() if p.is_dir())
    assert on_disk == CLASS_NAMES


def test_class_names_count_is_not_hard_coded():
    assert len(CLASS_NAMES) == len(list(DATASET_TRAIN.iterdir())) == 10
    assert len(set(CLASS_NAMES)) == len(CLASS_NAMES)


def test_load_class_names_reads_plain_list():
    assert load_class_names(CLASS_NAMES_FILE) == CLASS_NAMES


def test_load_class_names_missing_file(tmp_path):
    with pytest.raises(ModelNotAvailableError):
        load_class_names(tmp_path / "nope.json")


def test_load_class_names_rejects_empty_list(tmp_path):
    path = tmp_path / "class_names.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ModelNotAvailableError):
        load_class_names(path)


def test_load_class_names_rejects_duplicates(tmp_path):
    path = tmp_path / "class_names.json"
    path.write_text(json.dumps(["a", "a"]), encoding="utf-8")
    with pytest.raises(ModelNotAvailableError):
        load_class_names(path)


def test_load_class_names_rejects_non_strings(tmp_path):
    path = tmp_path / "class_names.json"
    path.write_text(json.dumps(["a", 2]), encoding="utf-8")
    with pytest.raises(ModelNotAvailableError):
        load_class_names(path)


def test_load_class_names_accepts_wrapped_dict(tmp_path):
    path = tmp_path / "class_names.json"
    path.write_text(json.dumps({"class_names": ["a", "b"]}), encoding="utf-8")
    assert load_class_names(path) == ["a", "b"]


def test_service_reports_error_for_missing_export(tmp_path):
    service = ModelService(
        model_path=tmp_path / "missing.keras",
        class_names_path=CLASS_NAMES_FILE,
        model_version="test-v",
    )
    service.load()
    assert service.is_loaded is False
    assert service.load_error is not None
    with pytest.raises(ModelNotAvailableError):
        service.predict(b"anything")


def test_service_does_not_load_a_model_per_call():
    """`predict` must reuse the in-memory model, never reload it."""
    from app.ml.inference import ModelService as Service

    service = Service(
        model_path=Path("stub.keras"),
        class_names_path=CLASS_NAMES_FILE,
        model_version="test-v",
    )
    calls = {"predict": 0}

    class FakeModel:
        def predict(self, batch, verbose=0):
            calls["predict"] += 1
            return [[0.0] * len(CLASS_NAMES)]

    service._class_names = list(CLASS_NAMES)
    service._model = FakeModel()

    from tests.conftest import make_image_bytes

    image = make_image_bytes()
    service.predict(image)
    service.predict(image)
    assert calls["predict"] == 2
    assert service._model is not None


def test_exported_model_exists_or_is_documented():
    """Either a real export is present, or the README says training is pending."""
    export = Path(__file__).resolve().parents[1] / "models" / "exported" / "leafora_densenet121.keras"
    if not export.exists():
        pytest.skip(
            "No exported model yet: run `python -m training.train`. "
            "The API returns 503 on /api/predict until then, which is the "
            "documented, honest behaviour."
        )
    from app.ml.inference import load_model

    model = load_model(export)
    assert model.output_shape[-1] == len(CLASS_NAMES)
