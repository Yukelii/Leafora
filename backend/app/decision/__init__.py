"""Rule-based decision layer.

This package is deliberately separate from the image classifier:

* ``symptoms`` holds the user-answered symptom checklist and its scoring.
* ``diagnosis`` combines the model's photo probabilities with that score and
  decides whether the answer is trustworthy enough to show as a diagnosis.

The symptom weights below are hand-assigned heuristics taken from published
tomato disease descriptions (see ``symptoms.EVIDENCE_NOTE``). They are **not**
learnt, **not** clinically validated, and must never be described as a trained
model or as proof of a disease.
"""

from app.decision.diagnosis import DiagnosisEngine
from app.decision.symptoms import QUESTIONS, score_answers

__all__ = ["DiagnosisEngine", "QUESTIONS", "score_answers"]
