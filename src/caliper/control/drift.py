"""Step size as a feedback loop on realised policy drift.

Drift is quadratic in the step size, so a multiplicative controller with exponent 1/2 corrects a
mis-set step in one update when the estimate is clean, and damps sensibly when it is not. The
controller needs only the drift a trainer already measures; no curvature and no Fisher.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DriftController:
    target: float
    step_size: float
    gain: float = 0.5
    max_ratio: float = 2.0
    smoothing: float = 0.0
    _filtered: float | None = field(default=None, repr=False)

    def update(self, realised_drift: float) -> float:
        """Consume the drift of the step just taken and return the next step size."""
        if realised_drift <= 0:
            return self.step_size
        if self.smoothing > 0:
            if self._filtered is None:
                self._filtered = realised_drift
            else:
                self._filtered += self.smoothing * (realised_drift - self._filtered)
            observed = self._filtered
        else:
            observed = realised_drift
        ratio = (self.target / observed) ** self.gain
        ratio = min(max(ratio, 1.0 / self.max_ratio), self.max_ratio)
        self.step_size *= ratio
        return self.step_size
