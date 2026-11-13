"""Tests for the rule-based decision layer.

These prove behaviour, not accuracy: the engine must refuse to answer when the
photo is weak, must surface disagreement instead of picking a side, and must
never leave the ten trained classes. The climate prior tests prove the same
thing for the sensor history: it may nudge, never outvote.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.decision.diagnosis import DiagnosisEngine, DiagnosisSettings
from app.decision.environment import EnvironmentSettings, build_prior
from app.schemas.sensor import SensorReading
from app.decision.symptoms import (
    ALL_CLASSES,
    QUESTIONS,
    QUESTIONS_BY_KEY,
    UnknownAnswerError,
    score_answers,
)

STRONG = {"Tomato___Late_blight": 0.93, "Tomato___Leaf_Mold": 0.03, "Tomato___healthy": 0.01}


def flat(**overrides):
    """A probability map over all ten classes with a chosen winner."""
    scores = {name: 0.005 for name in ALL_CLASSES}
    scores.update(overrides)
    total = sum(scores.values())
    return {name: value / total for name, value in scores.items()}


def test_strong_photo_without_answers_is_confident():
    engine = DiagnosisEngine()
    result = engine.evaluate(flat(**STRONG))
    assert result.status == "confident"
    assert result.predicted_class == "Tomato___Late_blight"
    assert result.confidence > 0.75


def test_low_confidence_photo_is_uncertain_and_names_no_class():
    engine = DiagnosisEngine()
    result = engine.evaluate(
        flat(**{"Tomato___Tomato_Yellow_Leaf_Curl_Virus": 0.61, "Tomato___Leaf_Mold": 0.2})
    )
    assert result.status == "uncertain"
    assert result.predicted_class is None
    assert "Retake" in result.message
    assert result.reasons


def test_narrow_margin_between_two_classes_is_uncertain():
    engine = DiagnosisEngine()
    result = engine.evaluate(
        flat(**{"Tomato___Early_blight": 0.4, "Tomato___Septoria_leaf_spot": 0.38})
    )
    assert result.status == "uncertain"


def test_photo_and_symptoms_disagree_is_reported_not_resolved():
    engine = DiagnosisEngine()
    # Photo says a virus, the checklist says Leaf Mold.
    answers = {"damage_type": "fuzzy", "underside": "grey_violet"}
    result = engine.evaluate(
        flat(**{"Tomato___Tomato_Yellow_Leaf_Curl_Virus": 0.9}), answers
    )
    assert result.status == "disagree"
    assert result.predicted_class is None
    assert "Leaf_Mold" in result.message.replace("___", "_")
    assert result.symptom_top is not None


def test_photo_and_symptoms_agree_is_confident():
    engine = DiagnosisEngine()
    answers = {"damage_type": "fuzzy", "underside": "grey_violet"}
    result = engine.evaluate(flat(**{"Tomato___Leaf_Mold": 0.9}), answers)
    assert result.status == "confident"
    assert result.predicted_class == "Tomato___Leaf_Mold"


def test_weights_are_reported_so_the_blend_is_visible():
    engine = DiagnosisEngine(DiagnosisSettings(image_weight=0.7, symptom_weight=0.3))
    result = engine.evaluate(flat(**STRONG), {"damage_type": "no_damage"})
    assert result.weights == {"photo": 0.7, "symptoms": 0.3}


def test_top_predictions_are_ranked_and_limited():
    engine = DiagnosisEngine(DiagnosisSettings(top_k=3))
    result = engine.evaluate(flat(**STRONG))
    assert len(result.top) == 3
    scores = [item.score for item in result.top]
    assert scores == sorted(scores, reverse=True)


def test_unknown_classes_in_model_output_are_ignored():
    engine = DiagnosisEngine()
    result = engine.evaluate({**flat(**STRONG), "Tomato___Not_A_Real_Class": 0.99})
    assert result.predicted_class == "Tomato___Late_blight"


def test_empty_probabilities_are_rejected():
    with pytest.raises(ValueError):
        DiagnosisEngine().evaluate({})


def test_maths_in_a_class_name_never_reaches_the_verdict():
    engine = DiagnosisEngine()
    result = engine.evaluate(
        flat(**{"Tomato___Leaf_Mold": 0.9}),
        {"leaf_curling": "upward"},
    )
    assert "Traceback" not in result.message


# -- checklist scoring -------------------------------------------------------


def test_scores_cover_exactly_the_ten_trained_classes():
    scores = score_answers({"damage_type": "dark_spots"})
    assert set(scores) == set(ALL_CLASSES)
    assert pytest.approx(sum(scores.values()), abs=1e-9) == 1.0


def test_no_useful_answers_give_a_uniform_distribution():
    scores = score_answers({"spot_centre": "na", "yellow_halo": "na"})
    assert len(set(round(value, 12) for value in scores.values())) == 1


def test_every_question_has_unique_keys_and_at_least_two_options():
    keys = [question.key for question in QUESTIONS]
    assert len(keys) == len(set(keys))
    for question in QUESTIONS:
        assert len(question.options) >= 2
        option_keys = [option.key for option in question.options]
        assert len(option_keys) == len(set(option_keys))


def test_unknown_question_key_is_an_error():
    with pytest.raises(UnknownAnswerError):
        score_answers({"what_colour_is_the_sky": "blue"})


def test_unknown_option_key_is_an_error():
    with pytest.raises(UnknownAnswerError):
        score_answers({"damage_type": "purple_polka_dots"})


def test_every_option_key_in_the_api_docs_exists():
    for question in QUESTIONS:
        for option in question.options:
            assert QUESTIONS_BY_KEY[question.key].option(option.key) is option


# -- climate prior from the sensor history -----------------------------------


def readings(hours: int, *, humidity: float, temperature: float, soil: float, step: int = 1):
    """One reading per ``step`` hours covering ``hours`` hours back from now."""
    now = datetime.now(timezone.utc)
    return [
        SensorReading(
            device_id="esp32-01",
            soil_moisture=soil,
            temperature=temperature,
            humidity=humidity,
            timestamp=now - timedelta(hours=offset),
        )
        for offset in range(hours, -1, -step)
    ]


def test_too_few_readings_give_no_prior():
    now = datetime.now(timezone.utc)
    pair = [
        SensorReading(
            device_id="esp32-01",
            soil_moisture=80,
            temperature=17,
            humidity=95,
            timestamp=now - timedelta(minutes=minutes),
        )
        for minutes in (5, 0)
    ]
    assert build_prior(pair, now=now) is None


def test_cool_wet_spell_favours_late_blight():
    prior = build_prior(readings(20, humidity=95, temperature=16, soil=80))
    assert prior is not None
    assert prior.favoured_class == "Tomato___Late_blight"
    assert prior.features.wet_hours > 15


def test_hot_dry_spell_favours_spider_mites():
    prior = build_prior(readings(20, humidity=40, temperature=34, soil=22))
    assert prior is not None
    assert prior.favoured_class == "Tomato___Spider_mites Two-spotted_spider_mite"


def test_prior_distribution_covers_the_ten_trained_classes_and_sums_to_one():
    prior = build_prior(readings(20, humidity=95, temperature=16, soil=80))
    assert prior is not None
    assert set(prior.distribution) == set(ALL_CLASSES)
    assert pytest.approx(sum(prior.distribution.values()), abs=1e-9) == 1.0


def test_readings_outside_the_window_are_ignored():
    now = datetime.now(timezone.utc)
    old = [
        SensorReading(
            device_id="esp32-01",
            soil_moisture=20,
            temperature=34,
            humidity=40,
            timestamp=now - timedelta(hours=hour),
        )
        for hour in range(72, 40, -1)
    ]
    assert build_prior(old, EnvironmentSettings(window_hours=24), now) is None


def test_climate_prior_raises_confidence_when_it_backs_the_photo():
    photo_answers = flat(**{"Tomato___Late_blight": 0.7, "Tomato___Leaf_Mold": 0.18})
    prior = build_prior(readings(20, humidity=95, temperature=16, soil=80))
    engine = DiagnosisEngine()
    plain = engine.evaluate(photo_answers)
    supported = engine.evaluate(photo_answers, None, prior)
    assert plain.status == "confident"
    assert supported.status == "confident"
    assert supported.confidence > plain.confidence
    assert "weights" in supported.to_dict()


def test_climate_prior_never_names_a_class_the_photo_did_not_see():
    # A very wet spell cannot turn a confident healthy-leaf photo into blight.
    engine = DiagnosisEngine()
    prior = build_prior(readings(20, humidity=95, temperature=16, soil=80))
    result = engine.evaluate(flat(**{"Tomato___healthy": 0.95}), None, prior)
    assert result.predicted_class == "Tomato___healthy"


def test_climate_weights_are_reported_when_a_prior_is_used():
    engine = DiagnosisEngine(DiagnosisSettings(climate_weight=0.2))
    prior = build_prior(readings(20, humidity=95, temperature=16, soil=80))
    result = engine.evaluate(flat(**{"Tomato___Late_blight": 0.93}), None, prior)
    assert pytest.approx(sum(result.weights.values()), abs=1e-9) == 1.0
    assert result.weights["climate"] == 0.2


def test_no_prior_keeps_the_old_weights_and_evidence():
    engine = DiagnosisEngine()
    result = engine.evaluate(flat(**STRONG), None, None)
    assert result.weights == {"photo": 1.0, "symptoms": 0.0}
    assert result.climate is None
