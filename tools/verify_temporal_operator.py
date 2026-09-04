"""Independent randomized verification for Step 4 S/S* and Tc/Tc*."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.temporal_operator import (
    BilinearTranslationOperator,
    MotionCorrectedTimeOperator,
)


def relative_dot_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-15)


def main() -> None:
    x = np.linspace(-15_000.0, 15_000.0, 31)
    y = np.linspace(-12_000.0, 12_000.0, 25)
    translation_errors = []
    time_errors = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        translations = [
            BilinearTranslationOperator(
                x, y,
                rng.uniform(-6000.0, 6000.0),
                rng.uniform(-6000.0, 6000.0),
            )
            for _ in range(4)
        ]
        source = rng.normal(size=(3, 4, y.size, x.size))
        dual = rng.normal(size=source.shape)
        left = float(np.vdot(translations[0].apply(source), dual))
        right = float(np.vdot(source, translations[0].adjoint(dual)))
        translation_errors.append(relative_dot_error(left, right))

        time_operator = MotionCorrectedTimeOperator(translations)
        time_source = rng.normal(size=(5, 3, 2, y.size, x.size))
        time_dual = rng.normal(size=(4, 3, 2, y.size, x.size))
        left = float(np.vdot(time_operator.apply(time_source), time_dual))
        right = float(np.vdot(time_source, time_operator.adjoint(time_dual)))
        time_errors.append(relative_dot_error(left, right))

    summary = {
        "translation_adjoint": {
            "seeds": len(translation_errors),
            "worst_relative_error": float(max(translation_errors)),
            "mean_relative_error": float(np.mean(translation_errors)),
            "acceptance_limit": 1e-7,
        },
        "time_operator_adjoint": {
            "seeds": len(time_errors),
            "worst_relative_error": float(max(time_errors)),
            "mean_relative_error": float(np.mean(time_errors)),
            "acceptance_limit": 1e-7,
        },
    }
    output_dir = ROOT / "artifacts" / "step4_temporal"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "operator_verification.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
