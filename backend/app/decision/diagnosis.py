"""Combine the photo prediction with the symptom checklist and sensor history.

Nothing here inspects the image, and nothing here changes the model output. It
only decides whether the answer is strong enough to present as a diagnosis,
and whether the independent sources agree.

The three inputs are deliberately separate
-------------------------------------------
``photo``
    What the classifier sees in the picture. Carries the most weight.
``symptoms``
    What the user reports on the checklist. Carries less.
``climate``
    What the ESP32 history says about the plant's weather (see
    ``app.decision.environment``). It is a *prior*: it can lift confidence when
    it backs the photo, and it can break a near-tie, but it can never outvote a
    confident photo.

Three outcomes
--------------
``confident``
    The winning class is clear in the photo (and, when answers were given, the
    checklist agrees). Present the disease.
``disagree``
    The photo and the checklist point at different diseases. Do not pick one:
    show both and ask for a closer photo.
``uncertain``
    The photo is weak (low confidence, or two classes too close) or the answers
    carried no usable information. Ask the user to retake the photo.

The gate exists because a softmax value is not a guarantee: a 0.61 top score on
an unseen photo style is the model saying "I am not sure", and the API should
repeat that instead of naming a disease confidently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.decision.environment import ENVIRONMENT_NOTE, EnvironmentPrior
from app.decision.symptoms import ALL_CLASSES, EVIDENCE_NOTE, score_answers


@dataclass(frozen=True)
class DiagnosisSettings:
    #: Weight of the photo probabilities when answers are supplied.
    image_weight: float = 0.6
    #: Weight of the symptom score when answers are supplied.
    symptom_weight: float = 0.4
    #: Share of the answer the climate prior may take. Deliberately small: the
    #: prior nudges the photo, it must never outvote it.
    climate_weight: float = 0.2
    #: Largest lift the climate prior may add to the photo confidence gate.
    climate_bonus: float = 0.08
    #: Below this top-1 probability the answer is reported as uncertain.
    min_confidence: float = 0.75
    #: Top-1 and top-2 closer than this means "cannot separate these two".
    min_margin: float = 0.15
    #: How many ranked classes to return.
    top_k: int = 3


@dataclass
class RankedClass:
    predicted_class: str
    score: float


@dataclass
class Diagnosis:
    status: str
    predicted_class: str | None
    confidence: float
    message: str
    top: list[RankedClass] = field(default_factory=list)
    photo_top: RankedClass | None = None
    symptom_top: RankedClass | None = None
    climate: EnvironmentPrior | None = None
    reasons: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "message": self.message,
            "top_predictions": [
                {"predicted_class": item.predicted_class, "score": round(item.score, 6)}
                for item in self.top
            ],
            "photo_prediction": (
                None
                if self.photo_top is None
                else {
                    "predicted_class": self.photo_top.predicted_class,
                    "confidence": round(self.photo_top.score, 6),
                }
            ),
            "symptom_prediction": (
                None
                if self.symptom_top is None
                else {
                    "predicted_class": self.symptom_top.predicted_class,
                    "score": round(self.symptom_top.score, 6),
                }
            ),
            "climate": (
                None
                if self.climate is None
                else {
                    "favoured_class": self.climate.favoured_class,
                    "support": round(self.climate.support, 6),
                    "features": self.climate.features.to_dict(),
                    "reasons": self.climate.reasons,
                }
            ),
            "reasons": self.reasons,
            "evidence": self.evidence,
            "weights": self.weights,
        }


def _rank(scores: dict[str, float], k: int) -> list[RankedClass]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [RankedClass(name, float(value)) for name, value in ordered[:k]]


class DiagnosisEngine:
    """Applies the confidence gate and the two-source agreement check."""

    def __init__(self, settings: DiagnosisSettings | None = None) -> None:
        self.settings = settings or DiagnosisSettings()

    def evaluate(
        self,
        photo_probabilities: dict[str, float],
        answers: dict[str, str] | None = None,
        climate: EnvironmentPrior | None = None,
    ) -> Diagnosis:
        """Weigh the photo (and optional answers and climate prior) into one answer.

        ``climate`` is the sensor-history prior. It can raise confidence when it
        backs the photo, and it can break a near-tie between the top two
        classes, but it never overrides a confident photo.
        """
        if not photo_probabilities:
            raise ValueError("photo_probabilities must not be empty.")

        # Only classes the model actually knows can be considered.
        photo = {name: float(photo_probabilities.get(name, 0.0)) for name in ALL_CLASSES}
        photo_ranked = _rank(photo, self.settings.top_k)
        photo_top = photo_ranked[0]
        photo_second = photo_ranked[1] if len(photo_ranked) > 1 else RankedClass("", 0.0)
        margin = photo_top.score - photo_second.score
        reasons: list[str] = []
        evidence: list[str] = [EVIDENCE_NOTE]
        if climate is not None:
            evidence.append(ENVIRONMENT_NOTE)

        symptom_scores: dict[str, float] | None = None
        symptom_top: RankedClass | None = None
        if answers:
            symptom_scores = score_answers(answers)
            symptom_top = _rank(symptom_scores, 1)[0]

        climate_weight = self.settings.climate_weight if climate is not None else 0.0
        # The photo and symptom weights share what the climate prior leaves.
        scale = 1.0 - climate_weight
        weights: dict[str, float] = (
            {
                "photo": self.settings.image_weight * scale,
                "symptoms": self.settings.symptom_weight * scale,
            }
            if symptom_scores is not None
            else {"photo": scale, "symptoms": 0.0}
        )
        if climate is not None:
            weights["climate"] = climate_weight

        # The photo and checklist score per class, without the climate share.
        # With no sensor history it equals the blend below.
        if symptom_scores is not None:
            base = {
                name: self.settings.image_weight * photo[name]
                + self.settings.symptom_weight * symptom_scores[name]
                for name in ALL_CLASSES
            }
        else:
            base = photo

        combined = {
            name: weights["photo"] * photo[name]
            + weights["symptoms"] * (symptom_scores[name] if symptom_scores else 0.0)
            + weights.get("climate", 0.0)
            * (climate.distribution[name] if climate is not None else 0.0)
            for name in ALL_CLASSES
        }

        combined_ranked = _rank(combined, self.settings.top_k)
        winner = combined_ranked[0]

        # The climate prior may lift the photo, but only when it backs the same
        # class and it actually found favourable weather.
        bonus = 0.0
        backs_photo = False
        if climate is not None and climate.favoured_class == photo_top.predicted_class:
            backs_photo = True
            bonus = self.settings.climate_bonus * min(1.0, max(0.0, climate.support))
            reasons.append(
                f"Sensor history backs {photo_top.predicted_class.replace('Tomato___', '')} "
                f"({climate.features.wet_hours:.1f} wet hours, "
                f"{climate.features.mean_temperature:.1f} C)."
            )
        elif climate is not None:
            reasons.append(
                f"Sensor history favours "
                f"{climate.favoured_class.replace('Tomato___', '')}, not the photo's choice."
            )

        # The number reported to the user: the winner's photo and checklist score,
        # lifted by the climate bonus when the sensor history backs it. The climate
        # share only ranks the classes. It is not blended into this number, because
        # a spread-out 10-class prior would pull a confident photo down.
        lift = bonus if winner.predicted_class == photo_top.predicted_class else 0.0
        reported = min(1.0, base[winner.predicted_class] + lift)
        effective_confidence = min(1.0, photo_top.score + bonus)
        # A strong climate signal may break a near-tie between the top two, but
        # only when it names one of them.
        tie_broken_by_climate = False
        if climate is not None and margin < self.settings.min_margin:
            top_two = {photo_top.predicted_class, photo_second.predicted_class}
            if climate.favoured_class in top_two and climate.support >= 0.5:
                tie_broken_by_climate = True

        low_confidence = effective_confidence < self.settings.min_confidence
        narrow_margin = margin < self.settings.min_margin and not tie_broken_by_climate
        if tie_broken_by_climate:
            reasons.append(
                f"The top two classes are close; sensor history favours "
                f"{climate.favoured_class.replace('Tomato___', '')}."  # type: ignore[union-attr]
            )
        if low_confidence:
            reasons.append(
                f"Photo confidence {photo_top.score:.2f} is below the "
                f"{self.settings.min_confidence:.2f} threshold."
            )
        if narrow_margin:
            reasons.append(
                f"Top two classes are only {margin:.2f} apart "
                f"(threshold {self.settings.min_margin:.2f})."
            )

        disagrees = symptom_top is not None and symptom_top.predicted_class != photo_top.predicted_class
        if disagrees:
            reasons.append(
                "The photo and the symptom checklist point at different diseases."
            )

        if disagrees:
            status = "disagree"
            message = (
                f"Photo suggests {photo_top.predicted_class.replace('Tomato___', '')} "
                f"({photo_top.score:.0%}), but the symptoms suggest "
                f"{symptom_top.predicted_class.replace('Tomato___', '')} "
                f"({symptom_top.score:.0%}). Retake the photo closer and in daylight."
            )
            return Diagnosis(
                status=status,
                predicted_class=None,
                confidence=reported,
                message=message,
                top=combined_ranked,
                photo_top=photo_top,
                symptom_top=symptom_top,
                climate=climate,
                reasons=reasons,
                evidence=evidence,
                weights=weights,
            )

        if low_confidence or narrow_margin:
            return Diagnosis(
                status="uncertain",
                predicted_class=None,
                confidence=reported,
                message=(
                    "Not enough evidence for a diagnosis. Retake the photo with one "
                    "leaf filling the frame, in daylight, without flash."
                    + (
                        " Answering the symptom questions will make this more reliable."
                        if symptom_top is None
                        else ""
                    )
                ),
                top=combined_ranked,
                photo_top=photo_top,
                symptom_top=symptom_top,
                climate=climate,
                reasons=reasons,
                evidence=evidence,
                weights=weights,
            )

        message = f"Most likely {winner.predicted_class.replace('Tomato___', '')}."
        if symptom_top is not None:
            message += " The symptom checklist agrees."
            reasons.append("Photo and symptom checklist agree.")
        if backs_photo:
            message += " Recent sensor readings support this."
            if bonus > 0.0:
                reasons.append(
                    f"Confidence raised by {bonus:.2f} from the sensor history."
                )
        return Diagnosis(
            status="confident",
            predicted_class=winner.predicted_class,
            confidence=reported,
            message=message,
            top=combined_ranked,
            photo_top=photo_top,
            symptom_top=symptom_top,
            climate=climate,
            reasons=reasons,
            evidence=evidence,
            weights=weights,
        )


__all__ = ["Diagnosis", "DiagnosisEngine", "DiagnosisSettings", "RankedClass"]
