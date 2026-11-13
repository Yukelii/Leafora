"""Turn the ESP32 sensor history into a climate prior for the diagnosis.

Why this exists
---------------
The photo model answers "what does this leaf look like". The sensor history
answers a different question: "has this plant been sitting in weather that
favours a particular disease". That is a *prior*, not a vote: cool nights with
long wet periods favour late blight, warm humid air favours bacterial spot and
leaf mould, hot dry air favours spider mites. A prior can only nudge the photo
result; it can never outvote it.

What is measured
----------------
Only what the ESP32 reports (soil moisture, air temperature, humidity) and only
over the window of readings supplied. Two of the derived features matter more
than the raw numbers:

``wet_hours``
    Hours where relative humidity stayed above the wetness threshold. A single
    humid reading is noise; nine straight wet hours is a signal.

``moisture_trend``
    Change in soil moisture per hour. A falling trend means the plant is drying
    out (mites, early blight pressure); a rising trend means it is being
    over-watered (root and leaf disease pressure).

Honesty rules
-------------
* Every number below is a hand-assigned heuristic from published tomato disease
  descriptions and extension-service guidance. Nothing here is trained, and
  nothing here is reported as an accuracy.
* Fewer readings, or readings spilling over a very short window, give no prior
  at all instead of a confident one.
* The prior distribution is blended with a small weight and is always reported
  in ``weights`` so the blend stays visible in the API response.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from app.decision.symptoms import (
    ALL_CLASSES,
    BACT,
    EARLY,
    HEALTHY,
    LATE,
    MITE,
    MOLD,
    MOSAIC,
    SEPT,
    TARGET,
    YLCV,
)
from app.schemas.sensor import SensorReading

ENVIRONMENT_NOTE = (
    "The climate prior is a hand-assigned heuristic built from published tomato "
    "disease descriptions and extension-service guidance (leaf wetness, humidity "
    "and temperature ranges). It is not trained on data and it is never reported "
    "as an accuracy."
)


@dataclass(frozen=True)
class EnvironmentSettings:
    """Thresholds and the shape of the prior.

    The values are working assumptions for the tropic / greenhouse conditions
    the project targets. They live here so they can be tuned once real field
    readings exist, without touching the decision engine.
    """

    #: Readings needed before any prior is produced at all.
    min_readings: int = 3
    #: A shorter window than this is "too little time" to mean anything.
    min_window_hours: float = 0.5
    #: Longest window considered; older readings age out of the picture.
    window_hours: float = 24.0
    #: Relative humidity (% RH) counted as leaf wetness.
    wet_humidity: float = 85.0
    #: Soil moisture (%) below this counts as dry soil.
    dry_moisture: float = 35.0
    #: Soil moisture (%) above this counts as over-wet soil.
    wet_moisture: float = 70.0
    #: How strongly the prior can pull a class (0 = no effect, 1 = doubles it).
    strength: float = 0.35
    #: Prior is only used when this much discriminating signal exists.
    min_signal: float = 0.08


@dataclass(frozen=True)
class MicroclimateFeatures:
    """What the sensor history says, in numbers a person can check."""

    sample_count: int
    window_hours: float
    mean_humidity: float
    wet_hours: float
    mean_temperature: float
    max_temperature: float
    min_temperature: float
    mean_soil_moisture: float
    moisture_trend: float

    @property
    def wet_fraction(self) -> float:
        return self.wet_hours / self.window_hours if self.window_hours > 0 else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sample_count": self.sample_count,
            "window_hours": round(self.window_hours, 2),
            "mean_humidity": round(self.mean_humidity, 2),
            "wet_hours": round(self.wet_hours, 2),
            "mean_temperature": round(self.mean_temperature, 2),
            "max_temperature": round(self.max_temperature, 2),
            "min_temperature": round(self.min_temperature, 2),
            "mean_soil_moisture": round(self.mean_soil_moisture, 2),
            "moisture_trend": round(self.moisture_trend, 2),
        }


@dataclass(frozen=True)
class EnvironmentPrior:
    """A probability-like distribution over the ten classes, plus the reasons."""

    features: MicroclimateFeatures
    distribution: dict[str, float]
    favoured_class: str
    support: float
    reasons: list[str]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ramp(value: float, low: float, high: float) -> float:
    """0.0 at ``low``, 1.0 at ``high``, linear in between."""
    if high == low:
        return 1.0 if value >= high else 0.0
    return _clamp01((value - low) / (high - low))


def features_from_readings(
    readings: Sequence[SensorReading],
    settings: EnvironmentSettings | None = None,
    now: datetime | None = None,
) -> MicroclimateFeatures | None:
    """Summarise the recent readings, or ``None`` when there is too little data.

    Readings older than ``window_hours`` (measured back from ``now``, or from the
    newest reading when ``now`` is not given) are dropped, and duplicate
    timestamps never count twice.
    """
    settings = settings or EnvironmentSettings()
    ordered = sorted(
        (reading for reading in readings if reading is not None),
        key=lambda reading: reading.timestamp,
    )
    if not ordered:
        return None

    anchor = now or ordered[-1].timestamp
    cutoff_seconds = settings.window_hours * 3600.0
    kept = [
        reading
        for reading in ordered
        if (anchor - reading.timestamp).total_seconds() <= cutoff_seconds
        and (anchor - reading.timestamp).total_seconds() >= 0
    ]
    if len(kept) < settings.min_readings:
        return None

    first, last = kept[0], kept[-1]
    window = (last.timestamp - first.timestamp).total_seconds() / 3600.0
    if window < settings.min_window_hours:
        return None

    # Leaf wetness: hours hosted above the threshold, weighted by the gap to the
    # next reading so an hourly ESP32 and a five-minute one agree.
    wet_hours = 0.0
    moisture_first = first.soil_moisture
    for current, following in zip(kept, kept[1:]):
        gap = (following.timestamp - current.timestamp).total_seconds() / 3600.0
        if current.humidity >= settings.wet_humidity:
            wet_hours += gap

    # The last sample contributes the gap to the anchor, so a recent dry-out is
    # counted and the window matches ``window`` in total.
    tail_gap = (anchor - last.timestamp).total_seconds() / 3600.0
    if tail_gap > 0 and last.humidity >= settings.wet_humidity:
        wet_hours += min(tail_gap, settings.window_hours)

    humidities = [reading.humidity for reading in kept]
    temperatures = [reading.temperature for reading in kept]
    moistures = [reading.soil_moisture for reading in kept]

    return MicroclimateFeatures(
        sample_count=len(kept),
        window_hours=window,
        mean_humidity=sum(humidities) / len(humidities),
        wet_hours=wet_hours,
        mean_temperature=sum(temperatures) / len(temperatures),
        max_temperature=max(temperatures),
        min_temperature=min(temperatures),
        mean_soil_moisture=sum(moistures) / len(moistures),
        # Positive means the soil is drying out over the window.
        moisture_trend=(moistures[-1] - moisture_first) / window if window > 0 else 0.0,
    )


def _class_fit(features: MicroclimateFeatures, settings: EnvironmentSettings) -> dict[str, float]:
    """A 0..1 fit per class: how well the measured climate suits that disease."""
    wet = _ramp(features.wet_fraction, 0.15, 0.70)
    # Cool: 1.0 at 18 C and below, 0.0 at 26 C and above.
    cool = _ramp(-features.mean_temperature, -26.0, -18.0)
    # Warm: 0.0 at 22 C, 1.0 at 28 C and above.
    warm = _ramp(features.mean_temperature, 22.0, 28.0)
    hot = _ramp(features.max_temperature, 30.0, 36.0)
    dry_soil = _ramp(-features.mean_soil_moisture, -settings.dry_moisture, -20.0)
    wet_soil = _ramp(features.mean_soil_moisture, settings.wet_moisture - 20.0, settings.wet_moisture + 15.0)
    drying = _ramp(features.moisture_trend, 0.5, 3.0)
    # Dry air: 1.0 at 40% RH and below, 0.0 at 60% RH and above.
    dry_air = _ramp(-features.mean_humidity, -60.0, -40.0)
    # Mild greenhouse air: 1.0 between 20 and 26 C, tapering outside.
    mild = min(
        _ramp(features.mean_temperature, 16.0, 20.0),
        _ramp(-features.mean_temperature, -30.0, -26.0),
    )

    return {
        # Cool and long wet periods: the classic late blight weather.
        LATE: 0.6 * cool + 0.3 * wet + 0.1 * wet_soil,
        # High humidity with mild air, typical of a covered or crowded canopy.
        MOLD: 0.6 * wet + 0.3 * mild + 0.1 * wet_soil,
        # Warm and wet: bacterial spot and its splash dispersal.
        BACT: 0.5 * warm + 0.4 * wet + 0.1 * wet_soil,
        # Warm, humid, alternating wet and dry leaves.
        SEPT: 0.4 * wet + 0.35 * warm + 0.25 * wet_soil,
        # Warm with drying soil: early blight and stressed older leaves.
        EARLY: 0.45 * warm + 0.3 * drying + 0.25 * dry_soil,
        # Warm and humid, like the others, but it tolerates drier soil.
        TARGET: 0.45 * warm + 0.4 * wet + 0.15 * mild,
        # Hot and dry air: the two-spotted spider mite's favourite weather.
        MITE: 0.45 * hot + 0.3 * dry_air + 0.15 * dry_soil + 0.1 * drying,
        # Hot and dry also drives whitefly, the yellow leaf curl vector.
        YLCV: 0.6 * hot + 0.25 * dry_soil + 0.15 * drying,
        # Mosaic virus spreads by contact and seed, so climate barely matters.
        MOSAIC: 0.5 * (warm + cool),
        # A healthy plant is the one outside every favourable window above.
        HEALTHY: 0.5 * (1.0 - max(wet, warm, hot)),
    }


def build_prior(
    readings: Sequence[SensorReading],
    settings: EnvironmentSettings | None = None,
    now: datetime | None = None,
) -> EnvironmentPrior | None:
    """Build the climate prior, or ``None`` when the history carries no signal.

    ``None`` is the honest answer for "two readings ten minutes apart" and for
    "readings in the middle of every comfortable range": a neutral distribution
    blended into the answer would only add noise.
    """
    settings = settings or EnvironmentSettings()
    features = features_from_readings(readings, settings, now)
    if features is None:
        return None

    fits = _class_fit(features, settings)
    fits = {name: _clamp01(fits.get(name, 0.5)) for name in ALL_CLASSES}
    signal = max(fits.values()) - min(fits.values())
    if signal < settings.min_signal:
        return None

    # Bias every class towards its fit, then normalise. A class that suits the
    # climate gains at most ``strength`` extra share, so the prior nudges.
    biased = {
        name: (1.0 / len(ALL_CLASSES)) * (1.0 + settings.strength * (value - 0.5))
        for name, value in fits.items()
    }
    total = sum(biased.values())
    distribution = {name: value / total for name, value in biased.items()}

    favoured = max(fits.items(), key=lambda item: item[1])
    lab = {
        LATE: "a cool, wet spell",
        MOLD: "humid, mild air",
        BACT: "warm, wet weather",
        SEPT: "warm, humid weather with wet soil",
        EARLY: "warm weather with drying soil",
        TARGET: "warm, humid weather",
        MITE: "hot, dry weather",
        YLCV: "hot, dry weather",
        MOSAIC: "no specific climate signal",
        HEALTHY: "no disease-favourable weather",
    }
    reasons = []
    if favoured[1] >= 0.5:
        reasons.append(
            f"Sensor history ({features.wet_hours:.1f} wet hours of "
            f"{features.window_hours:.1f} h, mean {features.mean_temperature:.1f} C, "
            f"{features.mean_humidity:.0f}% RH) looks like {lab.get(favoured[0], 'mixed weather')}, "
            f"which favours {favoured[0].replace('Tomato___', '')}."
        )
    else:
        reasons.append("Sensor history shows no climate clearly favouring a disease.")

    return EnvironmentPrior(
        features=features,
        distribution=distribution,
        favoured_class=favoured[0],
        support=float(favoured[1]),
        reasons=reasons,
    )


__all__ = [
    "ENVIRONMENT_NOTE",
    "EnvironmentPrior",
    "EnvironmentSettings",
    "MicroclimateFeatures",
    "build_prior",
    "features_from_readings",
]
