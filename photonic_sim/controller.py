from dataclasses import dataclass
from typing import Optional

import numpy as np

from .agent import AgentEnv
from .calibration import CalibrationBootstrapResult
from .inference import CalibrationState, estimate_resonances_from_osa


@dataclass(frozen=True)
class BootstrapRetuningControllerConfig:
    osa_span_nm: float = 0.4
    settle_ms: float = 12.0
    correction_gain: float = 0.6
    max_rounds: int = 4
    max_voltage_step_v: float = 0.5
    ridge: float = 1e-6


class BootstrapRetuningController:
    """Use calibration priors plus OSA scans to iteratively retune the array."""

    def __init__(
        self,
        calibration_result: Optional[CalibrationBootstrapResult] = None,
        calibration_state: Optional[CalibrationState] = None,
        config: Optional[BootstrapRetuningControllerConfig] = None,
        tuning_efficiency_nm_per_mw: float = 0.015,
        heater_resistance_ohm: float = 1200.0,
        max_voltage_v: float = 5.0,
    ):
        self.calibration_state = (
            calibration_state
            if calibration_state is not None
            else (
                None
                if calibration_result is None
                else CalibrationState.from_bootstrap_result(calibration_result)
            )
        )
        self.config = config or BootstrapRetuningControllerConfig()
        self.tuning_efficiency_nm_per_mw = float(
            tuning_efficiency_nm_per_mw
            if self.calibration_state is None
            else self.calibration_state.tuning_efficiency_nm_per_mw
        )
        self.heater_resistance_ohm = float(heater_resistance_ohm)
        self.max_voltage_v = float(max_voltage_v)

    def run_episode(self, env: AgentEnv) -> dict:
        if env.runtime is None:
            raise RuntimeError("env.reset() must be called before run_episode()")

        commanded_voltages_v = env.observation().commanded_voltages_v.copy()
        osa_frame_period_ms = 0.0 if env.runtime.osa is None else float(env.runtime.osa.config.frame_period_ms)

        for _ in range(self.config.max_rounds):
            if (
                osa_frame_period_ms > 0.0
                and env.observation().latest_osa_frame is not None
                and env.runtime.plant.time_ms == env.observation().latest_osa_frame.timestamp_ms
            ):
                step_result = env.step({"type": "wait", "dt_ms": osa_frame_period_ms})
                if step_result.done:
                    return step_result.info

            center_nm = float(np.mean(env.task_spec.target_resonances_nm))
            target_span_nm = float(
                np.max(env.task_spec.target_resonances_nm) - np.min(env.task_spec.target_resonances_nm)
            )
            step_result = env.step(
                {
                    "type": "read_osa",
                    "center_nm": center_nm,
                    "span_nm": max(self.config.osa_span_nm, target_span_nm + self.config.osa_span_nm),
                }
            )
            if step_result.done:
                return step_result.info

            observed_resonances_nm = self._estimate_resonances(
                frame=step_result.observation.latest_osa_frame,
                target_resonances_nm=env.task_spec.target_resonances_nm,
            )
            error_nm = observed_resonances_nm - env.task_spec.target_resonances_nm
            if float(np.max(np.abs(error_nm)) * 1000.0) <= env.task_spec.tolerance_pm:
                break

            commanded_voltages_v = self._propose_voltages(
                commanded_voltages_v=commanded_voltages_v,
                error_nm=error_nm,
            )

            for channel, voltage_v in enumerate(commanded_voltages_v):
                step_result = env.step(
                    {"type": "set_voltage", "channel": int(channel), "voltage_v": float(voltage_v)}
                )
                if step_result.done:
                    return step_result.info

            step_result = env.step({"type": "wait", "dt_ms": max(self.config.settle_ms, osa_frame_period_ms)})
            if step_result.done:
                return step_result.info

        return env.step({"type": "finish"}).info

    def _propose_voltages(self, commanded_voltages_v: np.ndarray, error_nm: np.ndarray) -> np.ndarray:
        num_rings = error_nm.shape[0]
        crosstalk_matrix = self._crosstalk_matrix(num_rings)
        regularized = crosstalk_matrix + self.config.ridge * np.eye(num_rings)
        delta_self_shift_nm = np.linalg.pinv(regularized) @ (-self.config.correction_gain * error_nm)
        current_power_mw = (np.asarray(commanded_voltages_v, dtype=float) ** 2) / self.heater_resistance_ohm * 1000.0
        delta_power_mw = delta_self_shift_nm / max(self.tuning_efficiency_nm_per_mw, 1e-12)
        next_power_mw = np.clip(
            current_power_mw + delta_power_mw,
            0.0,
            (self.max_voltage_v ** 2) / self.heater_resistance_ohm * 1000.0,
        )
        next_voltages_v = np.sqrt(next_power_mw * self.heater_resistance_ohm / 1000.0)
        clipped_delta_v = np.clip(
            next_voltages_v - np.asarray(commanded_voltages_v, dtype=float),
            -self.config.max_voltage_step_v,
            self.config.max_voltage_step_v,
        )
        return np.clip(np.asarray(commanded_voltages_v, dtype=float) + clipped_delta_v, 0.0, self.max_voltage_v)

    def _crosstalk_matrix(self, num_rings: int) -> np.ndarray:
        if self.calibration_state is None:
            return np.eye(num_rings, dtype=float)
        return self.calibration_state.for_num_rings(num_rings).crosstalk_matrix.copy()

    def _estimate_resonances(self, frame, target_resonances_nm: np.ndarray) -> np.ndarray:
        return estimate_resonances_from_osa(
            frame,
            reference_resonances_nm=target_resonances_nm,
            local_window_nm=self.config.osa_span_nm,
        )
