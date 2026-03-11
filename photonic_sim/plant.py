from typing import Optional

import numpy as np

from .config import (
    ActionExecutorConfig,
    MRRPlantConfig,
    SafetyGuardConfig,
)
from .execution import ActionExecutor, SafetyGuard
from .physics import (
    build_crosstalk_matrix,
    fold_detuning,
    lorentzian_transmission,
)
from .types import ActionAck, LatentPlantState


class MRRArrayPlant:
    """Minimal latent-state plant for online retuning experiments."""

    def __init__(self, num_rings: int, comb_wavelengths_nm: np.ndarray,
                 config: Optional[MRRPlantConfig] = None,
                 rng: Optional[np.random.Generator] = None,
                 action_config: Optional[ActionExecutorConfig] = None,
                 safety_config: Optional[SafetyGuardConfig] = None):
        self.config = config or MRRPlantConfig()
        self.rng = rng or np.random.default_rng()
        self.executor = ActionExecutor(action_config)
        self.num_rings = num_rings
        self.comb_wavelengths_nm = np.asarray(comb_wavelengths_nm, dtype=float)[:num_rings].copy()
        if self.comb_wavelengths_nm.shape[0] != num_rings:
            raise ValueError("comb_wavelengths_nm must provide at least num_rings wavelengths")

        self.base_resonances_nm = self.comb_wavelengths_nm.copy()
        self.bandwidth_nm = self.base_resonances_nm / self.config.q_factor
        self.hwhm_nm = self.bandwidth_nm / 2.0
        self.min_t = 10 ** (-self.config.extinction_ratio_db / 10.0)
        self.crosstalk_matrix = build_crosstalk_matrix(
            num_rings,
            self.config.crosstalk_alpha,
            self.config.crosstalk_decay_length,
        )
        self.safety = SafetyGuard(
            safety_config,
            tuning_efficiency_nm_per_mw=self.config.tuning_efficiency_nm_per_mw,
        )

        self.time_ms = 0.0
        self.target_voltages_v = np.zeros(num_rings, dtype=float)
        self.actuator_voltages_v = np.zeros(num_rings, dtype=float)
        self.command_powers_mw = np.zeros(num_rings, dtype=float)
        self.thermal_powers_mw = np.zeros(num_rings, dtype=float)
        self.drift_nm = np.zeros(num_rings, dtype=float)
        self.global_temp_shift_nm = self.config.initial_global_temp_shift_nm
        self.ideal_shifts_nm = np.zeros(num_rings, dtype=float)
        self.effective_resonances_nm = self.base_resonances_nm.copy()
        self.self_shift_clamped_mask = np.zeros(num_rings, dtype=bool)
        self.total_shift_warning_mask = np.zeros(num_rings, dtype=bool)

    def issue_command(self, channel: int, target_voltage_v: float) -> ActionAck:
        if channel < 0 or channel >= self.num_rings:
            raise IndexError(f"channel {channel} out of range for {self.num_rings} rings")
        decision = self.executor.sanitize_voltage(float(target_voltage_v))
        if decision.status != "rejected":
            self.target_voltages_v[channel] = decision.accepted_voltage_v
        return ActionAck(
            channel=channel,
            requested_voltage_v=float(target_voltage_v),
            target_voltage_v=float(self.target_voltages_v[channel]),
            issued_at_ms=self.time_ms,
            status=decision.status,
            message=decision.message,
        )

    def set_global_temp_shift_nm(self, shift_nm: float) -> None:
        self.global_temp_shift_nm = float(shift_nm)
        self._recompute_latent_state()

    def step(self, dt_ms: float) -> LatentPlantState:
        if dt_ms < 0:
            raise ValueError("dt_ms must be non-negative")

        if dt_ms > 0:
            self.actuator_voltages_v = self.executor.slew_toward(
                self.actuator_voltages_v,
                self.target_voltages_v,
                dt_ms,
            )
            self.command_powers_mw = (
                (self.actuator_voltages_v ** 2) / self.config.heater_resistance_ohm * 1000.0
            )

            alpha = 1.0 - np.exp(-dt_ms / max(self.config.thermal_tau_ms, 1e-9))
            self.thermal_powers_mw += alpha * (self.command_powers_mw - self.thermal_powers_mw)
            self.thermal_powers_mw, self.self_shift_clamped_mask = self.safety.clip_thermal_power(
                self.thermal_powers_mw
            )

            dt_s = dt_ms / 1000.0
            if self.config.drift_sigma_nm_per_s > 0:
                self.drift_nm += self.rng.normal(
                    0.0,
                    self.config.drift_sigma_nm_per_s * np.sqrt(dt_s),
                    size=self.num_rings,
                )

        self.time_ms += dt_ms
        self._recompute_latent_state()
        return self.latent_state()

    def latent_state(self) -> LatentPlantState:
        return LatentPlantState(
            time_ms=self.time_ms,
            actuator_voltages_v=self.actuator_voltages_v.copy(),
            target_voltages_v=self.target_voltages_v.copy(),
            command_powers_mw=self.command_powers_mw.copy(),
            thermal_powers_mw=self.thermal_powers_mw.copy(),
            ideal_shifts_nm=self.ideal_shifts_nm.copy(),
            drift_nm=self.drift_nm.copy(),
            effective_resonances_nm=self.effective_resonances_nm.copy(),
            global_temp_shift_nm=self.global_temp_shift_nm,
            self_shift_clamped_mask=self.self_shift_clamped_mask.copy(),
            total_shift_warning_mask=self.total_shift_warning_mask.copy(),
        )

    def per_ring_transmission(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
        delta = wavelengths_nm[None, :] - self.effective_resonances_nm[:, None]
        delta = fold_detuning(delta, self.config.fsr_nm)
        return lorentzian_transmission(delta, self.hwhm_nm[:, None], self.min_t)

    def total_through_transmission(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        return np.prod(self.per_ring_transmission(wavelengths_nm), axis=0)

    def comb_line_throughput(self, input_powers_mw: Optional[np.ndarray] = None) -> np.ndarray:
        transmission = self.total_through_transmission(self.comb_wavelengths_nm)
        if input_powers_mw is None:
            return transmission
        input_powers_mw = np.asarray(input_powers_mw, dtype=float)
        return input_powers_mw[: self.num_rings] * transmission

    def _recompute_latent_state(self) -> None:
        self.ideal_shifts_nm = self.config.tuning_efficiency_nm_per_mw * self.thermal_powers_mw
        total_shift_nm = (
            self.crosstalk_matrix @ self.ideal_shifts_nm
            + self.drift_nm
            + self.global_temp_shift_nm
        )
        self.total_shift_warning_mask = self.safety.total_shift_warning(total_shift_nm)
        self.effective_resonances_nm = self.base_resonances_nm + total_shift_nm
