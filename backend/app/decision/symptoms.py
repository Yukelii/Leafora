"""Symptom checklist: the questions and the scoring behind them.

How it works
------------
Each question has a fixed list of answers. Each answer carries points towards
the tomato diseases it is typical of. Choosing an answer adds those points, and
the totals are normalised into a probability-like distribution over the ten
trained classes.

Honesty rules
-------------
* Only the ten classes in ``app/ml/class_names.json`` can ever be scored, so the
  checklist can never name a disease the model was not trained on.
* The weights are expert-assigned from published descriptions, not learnt from
  data. They are reported as a *symptom score*, never as accuracy.
* Answers are free of typo-tolerance by design: an unknown key or option is a
  400/422 error, not a silent guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Class names exactly as trained (index order does not matter here, the name
# does: the engine maps these names onto the model's class list).
BACT = "Tomato___Bacterial_spot"
EARLY = "Tomato___Early_blight"
LATE = "Tomato___Late_blight"
MOLD = "Tomato___Leaf_Mold"
SEPT = "Tomato___Septoria_leaf_spot"
MITE = "Tomato___Spider_mites Two-spotted_spider_mite"
TARGET = "Tomato___Target_Spot"
YLCV = "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
MOSAIC = "Tomato___Tomato_mosaic_virus"
HEALTHY = "Tomato___healthy"

ALL_CLASSES: tuple[str, ...] = (
    BACT,
    EARLY,
    LATE,
    MOLD,
    SEPT,
    MITE,
    TARGET,
    YLCV,
    MOSAIC,
    HEALTHY,
)

EVIDENCE_NOTE = (
    "Symptom weights are hand-assigned from published tomato disease "
    "descriptions (extension service field guides). They are heuristics, not a "
    "trained or clinically validated diagnostic model."
)


@dataclass(frozen=True)
class Option:
    """One selectable answer and the diseases it points at."""

    key: str
    label: str
    weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Question:
    key: str
    prompt: str
    options: tuple[Option, ...]

    def option(self, key: str) -> Option | None:
        for candidate in self.options:
            if candidate.key == key:
                return candidate
        return None


QUESTIONS: tuple[Question, ...] = (
    Question(
        key="damage_type",
        prompt="What does the main damage on the leaf look like?",
        options=(
            Option(
                "dark_spots",
                "Dark or brown spots on the leaf",
                {BACT: 2, EARLY: 2, LATE: 1, SEPT: 2, TARGET: 2},
            ),
            Option(
                "pale_patches",
                "Pale yellow patches, not distinct spots",
                {YLCV: 2, MOSAIC: 1, MOLD: 1},
            ),
            Option("curling", "Leaves curled, cupped or twisted", {YLCV: 3, MOSAIC: 1}),
            Option("mosaic", "Mixed light and dark green patches", {MOSAIC: 3, YLCV: 1}),
            Option("fuzzy", "Fuzzy growth on the leaf", {MOLD: 3}),
            Option("stippling", "Fine pale speckling, leaf looks dusty", {MITE: 3}),
            Option("no_damage", "No damage visible", {HEALTHY: 5}),
        ),
    ),
    Question(
        key="spot_centre",
        prompt="If there are spots, what colour is the centre?",
        options=(
            Option("black_small", "Small black specks", {BACT: 3, SEPT: 1}),
            Option("brown", "Brown centre", {EARLY: 3, TARGET: 1}),
            Option("rings", "Concentric rings, like a target", {TARGET: 3, EARLY: 1}),
            Option("water_soaked", "Grey-green, water-soaked edge", {LATE: 3}),
            Option("white_centre", "Pale grey or white centre", {SEPT: 3}),
            Option("na", "Not applicable", {}),
        ),
    ),
    Question(
        key="yellow_halo",
        prompt="Do the spots have a yellow ring around them?",
        options=(
            Option("strong", "Yes, a clear yellow halo", {TARGET: 2, EARLY: 2, SEPT: 2, BACT: 1}),
            Option("mild", "Faint yellowing only", {SEPT: 1, MOLD: 1}),
            Option("none", "No yellow ring", {LATE: 2, MOLD: 1}),
            Option("na", "Not applicable", {}),
        ),
    ),
    Question(
        key="leaf_curling",
        prompt="Are the leaves curled or distorted?",
        options=(
            Option("upward", "Yes, edges curl upward or cup", {YLCV: 4}),
            Option("twisted", "Yes, twisted or narrowed leaflets", {MOSAIC: 2, YLCV: 1}),
            Option("mild", "Slightly", {MOSAIC: 1}),
            Option("none", "No curling", {}),
        ),
    ),
    Question(
        key="mottle",
        prompt="Is there a light-and-dark green mottled pattern?",
        options=(
            Option("yes_speckled", "Yes, speckled or blistered", {MOSAIC: 4}),
            Option("yes_broad", "Yes, broad yellow-green areas", {YLCV: 3}),
            Option("no", "No", {}),
        ),
    ),
    Question(
        key="underside",
        prompt="Any fuzzy growth under the leaf?",
        options=(
            Option("grey_violet", "Grey to violet fuzz", {MOLD: 4}),
            Option("white", "White fuzz or white mould", {LATE: 1, MOLD: 1}),
            Option("none", "None", {}),
            Option("not_checked", "Not checked", {}),
        ),
    ),
    Question(
        key="webbing",
        prompt="Any fine webbing or pale stippling?",
        options=(
            Option("webbing", "Yes, fine webbing", {MITE: 4}),
            Option("stippling", "Pale stippling, no webbing", {MITE: 2}),
            Option("no", "No", {}),
            Option("not_checked", "Not checked", {}),
        ),
    ),
    Question(
        key="first_leaves",
        prompt="Which leaves showed it first?",
        options=(
            Option(
                "lower",
                "Older, lower leaves",
                {EARLY: 1, SEPT: 2, LATE: 1, MOLD: 1, MITE: 1},
            ),
            Option("new", "New, upper growth", {YLCV: 2, MOSAIC: 2, BACT: 1}),
            Option("all", "Spread across the whole plant", {BACT: 1, LATE: 1}),
            Option("not_sure", "Not sure", {}),
        ),
    ),
)

QUESTIONS_BY_KEY: dict[str, Question] = {question.key: question for question in QUESTIONS}


class UnknownAnswerError(ValueError):
    """An answer key or option was not part of the checklist."""


def score_answers(answers: dict[str, str]) -> dict[str, float]:
    """Turn checklist answers into a score per trained class.

    Args:
        answers: ``{question_key: option_key}``. Questions may be omitted; an
            omitted question simply adds no points.

    Returns:
        ``{class_name: score}`` summing to 1.0, or a uniform distribution when
        no answer carried any weight (so "no information" never looks like
        a confident result).

    Raises:
        UnknownAnswerError: an unknown question key or option key.
    """
    if not isinstance(answers, dict):
        raise UnknownAnswerError("Answers must be an object of question -> option.")

    totals: dict[str, float] = {name: 0.0 for name in ALL_CLASSES}
    for question_key, option_key in answers.items():
        question = QUESTIONS_BY_KEY.get(question_key)
        if question is None:
            raise UnknownAnswerError(
                f"Unknown question '{question_key}'. "
                f"Valid questions: {', '.join(QUESTIONS_BY_KEY)}."
            )
        option = question.option(str(option_key))
        if option is None:
            raise UnknownAnswerError(
                f"Unknown option '{option_key}' for question '{question_key}'. "
                f"Valid options: {', '.join(o.key for o in question.options)}."
            )
        for class_name, points in option.weights.items():
            totals[class_name] += float(points)

    total = sum(totals.values())
    if total <= 0.0:
        uniform = 1.0 / len(ALL_CLASSES)
        return {name: uniform for name in ALL_CLASSES}
    return {name: value / total for name, value in totals.items()}


__all__ = [
    "ALL_CLASSES",
    "EVIDENCE_NOTE",
    "QUESTIONS",
    "QUESTIONS_BY_KEY",
    "Question",
    "Option",
    "UnknownAnswerError",
    "score_answers",
]
