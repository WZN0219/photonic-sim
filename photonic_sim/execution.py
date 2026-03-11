from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import ActionExecutorConfig, SafetyGuardConfig


@dataclass(frozen=True)
class CommandDecision:
    requested_voltage_v: float
    accepted_voltage_v: float
    status: str
    message: str


class ActionExecutor:
    def __init__(self, config: Optional[ActionExecutorConfig] = None):
        self.config = config or ActionExecutorConfig()

    def sanitize_voltage(self, requested_voltage_v: float) -> CommandDecision:
        if not np.isfinite(requested_voltage_v):
            return CommandDecision(
                requested_voltage_v=float(requested_voltage_v),
                accepted_voltage_v=0.0,
                status="rejected",
                message="non-finite voltage command",
            )

        clipped_voltage_v = float(np.clip(requested_voltage_v, 0.0, self.config.max_voltage_v))
        if clipped_voltage_v != float(requested_voltage_v):
            return CommandDecision(
                requested_voltage_v=float(requested_voltage_v),
                accepted_voltage_v=clipped_voltage_v,
                status="clamped",
                message=f"clamped to [0, {self.config.max_voltage_v}] V",
            )

        return CommandDecision(
            requested_voltage_v=float(requested_voltage_v),
            accepted_voltage_v=clipped_voltage_v,
            status="accepted",
            message="command accepted",
        )

    def slew_toward(self, current_voltage_v: np.ndarray, target_voltage_v: np.ndarray,
                    dt_ms: float) -> np.ndarray:
        max_delta_v = self.config.slew_rate_v_per_ms * dt_ms
        delta_v = np.clip(target_voltage_v - current_voltage_v, -max_delta_v, max_delta_v)
        return current_voltage_v + delta_v


class SafetyGuard:
    def __init__(self, config: Optional[SafetyGuardConfig] = None,
                 tuning_efficiency_nm_per_mw: float = 0.015):
        self.config = config or SafetyGuardConfig()
        self.tuning_efficiency_nm_per_mw = tuning_efficiency_nm_per_mw

    def clip_thermal_power(self, thermal_power_mw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        max_power_mw = self.config.max_self_shift_nm / self.tuning_efficiency_nm_per_mw
        clipped_power_mw = np.clip(thermal_power_mw, 0.0, max_power_mw)
        was_clipped = thermal_power_mw > max_power_mw
        return clipped_power_mw, was_clipped

    def total_shift_warning(self, total_shift_nm: np.ndarray) -> np.ndarray:
        return np.abs(total_shift_nm) > self.config.max_total_shift_nm
