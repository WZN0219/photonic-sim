from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from .agent import AgentObservation
from .calibration import CalibrationBootstrapResult
from .types import MeasurementFrame


def measurement_source_timestamp_ms(frame: Optional[MeasurementFrame]) -> Optional[float]:
    if frame is None:
        return None
    return float(frame.metadata.get("source_timestamp_ms", frame.timestamp_ms))


def crosstalk_profile_to_matrix(relative_profile_by_offset: dict[int, float], num_rings: int) -> np.ndarray:
    matrix = np.zeros((num_rings, num_rings), dtype=float)
    abs_profile = {abs(int(offset)): float(value) for offset, value in relative_profile_by_offset.items()}
    for i in range(num_rings):
        for j in range(num_rings):
            matrix[i, j] = float(abs_profile.get(abs(i - j), 0.0))
    return matrix


def estimate_resonances_from_osa(
    frame: MeasurementFrame,
    reference_resonances_nm: np.ndarray,
    local_window_nm: float,
) -> np.ndarray:
    reference_resonances_nm = np.asarray(reference_resonances_nm, dtype=float)
    wavelengths_nm = np.asarray(frame.payload["wavelengths_nm"], dtype=float)
    spectrum_dbm = np.asarray(frame.payload["spectrum_dbm"], dtype=float)

    if reference_resonances_nm.size > 1:
        sorted_refs = np.sort(reference_resonances_nm)
        half_window_nm = min(local_window_nm / 2.0, 0.45 * float(np.min(np.diff(sorted_refs))))
    else:
        half_window_nm = local_window_nm / 2.0

    observed_resonances_nm = []
    for reference_nm in reference_resonances_nm:
        mask = np.abs(wavelengths_nm - reference_nm) <= half_window_nm
        if not np.any(mask):
            nearest_idx = int(np.argmin(np.abs(wavelengths_nm - reference_nm)))
            observed_resonances_nm.append(float(wavelengths_nm[nearest_idx]))
            continue
        local_wavelengths_nm = wavelengths_nm[mask]
        local_spectrum_dbm = spectrum_dbm[mask]
        dip_idx = int(np.argmin(local_spectrum_dbm))
        observed_resonances_nm.append(float(local_wavelengths_nm[dip_idx]))
    return np.asarray(observed_resonances_nm, dtype=float)


@dataclass(frozen=True)
class CalibrationState:
    version: str
    source: str
    tuning_efficiency_nm_per_mw: float
    thermal_t63_ms: float
    thermal_t95_ms: float
    crosstalk_matrix: np.ndarray
    recommended_pd_config: dict[str, Any]
    recommended_osa_config: dict[str, Any]
    confidence: float = 1.0
    timestamp_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.crosstalk_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("crosstalk_matrix must be square")
        object.__setattr__(self, "crosstalk_matrix", matrix.copy())
        object.__setattr__(self, "recommended_pd_config", dict(self.recommended_pd_config))
        object.__setattr__(self, "recommended_osa_config", dict(self.recommended_osa_config))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_bootstrap_result(
        cls,
        result: CalibrationBootstrapResult,
        num_rings: Optional[int] = None,
        confidence: float = 1.0,
        timestamp_ms: float = 0.0,
    ) -> "CalibrationState":
        matrix_size = len(result.crosstalk.estimated_crosstalk_matrix)
        num_rings = matrix_size if num_rings is None else int(num_rings)
        crosstalk_matrix = crosstalk_profile_to_matrix(result.crosstalk.relative_profile_by_offset, num_rings)
        return cls(
            version=f"bootstrap:{Path(result.source_dir).name}",
            source=result.source_dir,
            tuning_efficiency_nm_per_mw=result.step_response.estimated_tuning_efficiency_nm_per_mw,
            thermal_t63_ms=result.step_response.t63_ms,
            thermal_t95_ms=result.step_response.t95_ms,
            crosstalk_matrix=crosstalk_matrix,
            recommended_pd_config=result.observation.recommended_pd_config,
            recommended_osa_config=result.observation.recommended_osa_config,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            timestamp_ms=float(timestamp_ms),
            metadata={
                "step_response_final_shift_nm": result.step_response.final_shift_nm,
                "drift_duration_ms": result.drift.duration_ms,
            },
        )

    def for_num_rings(self, num_rings: int) -> "CalibrationState":
        current_size = self.crosstalk_matrix.shape[0]
        if num_rings == current_size:
            return self
        profile_by_offset = {}
        for offset in range(current_size):
            diagonal = np.diag(self.crosstalk_matrix, k=offset)
            if diagonal.size == 0:
                continue
            profile_by_offset[offset] = float(np.mean(diagonal))
        rebuilt = crosstalk_profile_to_matrix(profile_by_offset, num_rings)
        return replace(self, crosstalk_matrix=rebuilt)

    def with_confidence(self, confidence: float, timestamp_ms: Optional[float] = None) -> "CalibrationState":
        next_timestamp_ms = self.timestamp_ms if timestamp_ms is None else float(timestamp_ms)
        return replace(
            self,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            timestamp_ms=next_timestamp_ms,
        )


@dataclass(frozen=True)
class BeliefState:
    time_ms: float
    resonance_estimates_nm: np.ndarray
    resonance_uncertainty_pm: np.ndarray
    identity_confidence: np.ndarray
    calibration_confidence: float
    last_pd_timestamp_ms: Optional[float] = None
    last_osa_timestamp_ms: Optional[float] = None
    last_observed_resonances_nm: Optional[np.ndarray] = None
    innovation_pm: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        estimates = np.asarray(self.resonance_estimates_nm, dtype=float)
        uncertainty = np.asarray(self.resonance_uncertainty_pm, dtype=float)
        confidence = np.asarray(self.identity_confidence, dtype=float)
        if estimates.ndim != 1 or uncertainty.shape != estimates.shape or confidence.shape != estimates.shape:
            raise ValueError("belief arrays must be 1D and share the same shape")
        object.__setattr__(self, "resonance_estimates_nm", estimates.copy())
        object.__setattr__(self, "resonance_uncertainty_pm", uncertainty.copy())
        object.__setattr__(self, "identity_confidence", np.clip(confidence, 0.0, 1.0))
        if self.last_observed_resonances_nm is not None:
            observed = np.asarray(self.last_observed_resonances_nm, dtype=float)
            if observed.shape != estimates.shape:
                raise ValueError("last_observed_resonances_nm must match resonance_estimates_nm")
            object.__setattr__(self, "last_observed_resonances_nm", observed.copy())
        if self.innovation_pm is not None:
            innovation = np.asarray(self.innovation_pm, dtype=float)
            if innovation.shape != estimates.shape:
                raise ValueError("innovation_pm must match resonance_estimates_nm")
            object.__setattr__(self, "innovation_pm", innovation.copy())
        object.__setattr__(self, "metadata", dict(self.metadata))


class BeliefStateEstimator(Protocol):
    def initialize(self, observation: AgentObservation, calibration_state: CalibrationState) -> BeliefState:
        ...

    def update(
        self,
        previous_belief: Optional[BeliefState],
        observation: AgentObservation,
        calibration_state: CalibrationState,
    ) -> BeliefState:
        ...


@dataclass(frozen=True)
class SimpleBeliefStateEstimatorConfig:
    osa_local_window_nm: float = 0.4
    initial_uncertainty_pm: float = 80.0
    fresh_osa_uncertainty_pm: float = 6.0
    stale_osa_uncertainty_pm: float = 18.0
    uncertainty_growth_pm_per_ms: float = 0.15
    saturation_uncertainty_boost_pm: float = 8.0
    identity_confidence_floor: float = 0.05
    divergence_scale_pm: float = 60.0
    calibration_decay_on_stale: float = 0.05
    calibration_decay_on_saturation: float = 0.15


class SimpleBeliefStateEstimator:
    def __init__(self, config: Optional[SimpleBeliefStateEstimatorConfig] = None):
        self.config = config or SimpleBeliefStateEstimatorConfig()

    def initialize(self, observation: AgentObservation, calibration_state: CalibrationState) -> BeliefState:
        num_rings = observation.task_spec.target_resonances_nm.shape[0]
        prior_belief = BeliefState(
            time_ms=observation.time_ms,
            resonance_estimates_nm=observation.task_spec.target_resonances_nm.copy(),
            resonance_uncertainty_pm=np.full(num_rings, self.config.initial_uncertainty_pm),
            identity_confidence=np.full(num_rings, calibration_state.confidence),
            calibration_confidence=calibration_state.confidence,
            metadata={"source": "prior"},
        )
        return self.update(prior_belief, observation, calibration_state)

    def update(
        self,
        previous_belief: Optional[BeliefState],
        observation: AgentObservation,
        calibration_state: CalibrationState,
    ) -> BeliefState:
        if previous_belief is None:
            return self.initialize(observation, calibration_state)

        dt_ms = max(float(observation.time_ms - previous_belief.time_ms), 0.0)
        estimates_nm = previous_belief.resonance_estimates_nm.copy()
        uncertainty_pm = previous_belief.resonance_uncertainty_pm + dt_ms * self.config.uncertainty_growth_pm_per_ms
        identity_confidence = np.clip(
            previous_belief.identity_confidence - 0.001 * dt_ms,
            self.config.identity_confidence_floor,
            1.0,
        )
        calibration_confidence = float(
            np.clip(min(previous_belief.calibration_confidence, calibration_state.confidence), 0.0, 1.0)
        )
        last_pd_timestamp_ms = previous_belief.last_pd_timestamp_ms
        last_osa_timestamp_ms = previous_belief.last_osa_timestamp_ms
        last_observed_resonances_nm = (
            None if previous_belief.last_observed_resonances_nm is None else previous_belief.last_observed_resonances_nm.copy()
        )
        innovation_pm = np.zeros_like(estimates_nm)
        metadata = {"source": "propagated"}

        osa_frame = observation.latest_osa_frame
        osa_measurement_timestamp_ms = measurement_source_timestamp_ms(osa_frame)
        if osa_frame is not None and (
            last_osa_timestamp_ms is None or osa_measurement_timestamp_ms > last_osa_timestamp_ms
        ):
            reference_resonances_nm = (
                observation.task_spec.target_resonances_nm
                if last_observed_resonances_nm is None
                else last_observed_resonances_nm
            )
            measured_resonances_nm = estimate_resonances_from_osa(
                osa_frame,
                reference_resonances_nm=reference_resonances_nm,
                local_window_nm=self.config.osa_local_window_nm,
            )
            innovation_pm = (measured_resonances_nm - estimates_nm) * 1000.0
            estimates_nm = measured_resonances_nm
            base_uncertainty_pm = (
                self.config.fresh_osa_uncertainty_pm
                if osa_frame.quality_flag == "fresh"
                else self.config.stale_osa_uncertainty_pm
            )
            uncertainty_pm = np.full_like(estimates_nm, base_uncertainty_pm) + 0.1 * np.abs(innovation_pm)
            identity_confidence = np.clip(
                1.0 - np.abs(innovation_pm) / max(self.config.divergence_scale_pm, 1e-9),
                self.config.identity_confidence_floor,
                1.0,
            )
            last_observed_resonances_nm = measured_resonances_nm
            last_osa_timestamp_ms = osa_measurement_timestamp_ms
            metadata["source"] = f"osa:{osa_frame.quality_flag}"
            if osa_frame.quality_flag == "stale":
                calibration_confidence -= self.config.calibration_decay_on_stale
        elif osa_frame is not None and osa_frame.quality_flag == "stale":
            calibration_confidence -= 0.5 * self.config.calibration_decay_on_stale

        pd_frame = observation.latest_pd_frame
        pd_measurement_timestamp_ms = measurement_source_timestamp_ms(pd_frame)
        if pd_measurement_timestamp_ms is not None and (
            last_pd_timestamp_ms is None or pd_measurement_timestamp_ms > last_pd_timestamp_ms
        ):
            last_pd_timestamp_ms = pd_measurement_timestamp_ms
        if pd_frame is not None and bool(pd_frame.metadata.get("saturated", False)):
            uncertainty_pm = uncertainty_pm + self.config.saturation_uncertainty_boost_pm
            calibration_confidence -= self.config.calibration_decay_on_saturation
            metadata["pd_saturated"] = True

        return BeliefState(
            time_ms=observation.time_ms,
            resonance_estimates_nm=estimates_nm,
            resonance_uncertainty_pm=np.maximum(uncertainty_pm, 1e-6),
            identity_confidence=identity_confidence,
            calibration_confidence=float(np.clip(calibration_confidence, 0.0, 1.0)),
            last_pd_timestamp_ms=last_pd_timestamp_ms,
            last_osa_timestamp_ms=last_osa_timestamp_ms,
            last_observed_resonances_nm=last_observed_resonances_nm,
            innovation_pm=innovation_pm,
            metadata=metadata,
        )


@dataclass(frozen=True)
class RecoveryConfig:
    stale_age_threshold_ms: float = 5.0
    belief_divergence_threshold_pm: float = 25.0
    collision_margin_pm: float = 30.0
    calibration_confidence_threshold: float = 0.35


@dataclass(frozen=True)
class RecoveryDecision:
    stale_measurement: bool
    saturation_detected: bool
    belief_divergence: bool
    collision_suspected: bool
    calibration_confidence_low: bool
    should_recover: bool
    suggested_action: str
    max_innovation_pm: float
    min_spacing_pm: float
    details: dict[str, Any] = field(default_factory=dict)


class RecoveryTrigger:
    def __init__(self, config: Optional[RecoveryConfig] = None):
        self.config = config or RecoveryConfig()

    def evaluate(
        self,
        observation: AgentObservation,
        belief_state: Optional[BeliefState],
        calibration_state: CalibrationState,
    ) -> RecoveryDecision:
        stale_age_ms = 0.0
        stale_measurement = False
        for frame in (observation.latest_pd_frame, observation.latest_osa_frame):
            if frame is None or frame.quality_flag != "stale":
                continue
            source_timestamp_ms = measurement_source_timestamp_ms(frame)
            stale_age_ms = max(stale_age_ms, float(frame.timestamp_ms - source_timestamp_ms))
            stale_measurement = stale_measurement or stale_age_ms >= self.config.stale_age_threshold_ms

        saturation_detected = bool(
            observation.latest_pd_frame is not None and observation.latest_pd_frame.metadata.get("saturated", False)
        )
        max_innovation_pm = 0.0
        min_spacing_pm = float("inf")
        belief_divergence = False
        collision_suspected = False
        calibration_confidence = calibration_state.confidence
        if belief_state is not None:
            calibration_confidence = min(calibration_confidence, belief_state.calibration_confidence)
            if belief_state.innovation_pm is not None:
                max_innovation_pm = float(np.max(np.abs(belief_state.innovation_pm)))
                belief_divergence = max_innovation_pm >= self.config.belief_divergence_threshold_pm
            if belief_state.resonance_estimates_nm.shape[0] > 1:
                sorted_estimates_nm = np.sort(belief_state.resonance_estimates_nm)
                min_spacing_pm = float(np.min(np.diff(sorted_estimates_nm)) * 1000.0)
                collision_suspected = min_spacing_pm <= self.config.collision_margin_pm

        calibration_confidence_low = calibration_confidence < self.config.calibration_confidence_threshold

        suggested_action = "none"
        if calibration_confidence_low:
            suggested_action = "rebootstrap"
        elif collision_suspected or belief_divergence:
            suggested_action = "global_osa_rescan"
        elif saturation_detected:
            suggested_action = "switch_to_osa_or_reduce_pd_gain"
        elif stale_measurement:
            suggested_action = "wait_for_fresh_measurement"

        should_recover = any(
            (
                stale_measurement,
                saturation_detected,
                belief_divergence,
                collision_suspected,
                calibration_confidence_low,
            )
        )
        return RecoveryDecision(
            stale_measurement=stale_measurement,
            saturation_detected=saturation_detected,
            belief_divergence=belief_divergence,
            collision_suspected=collision_suspected,
            calibration_confidence_low=calibration_confidence_low,
            should_recover=should_recover,
            suggested_action=suggested_action,
            max_innovation_pm=max_innovation_pm,
            min_spacing_pm=min_spacing_pm,
            details={
                "stale_age_ms": stale_age_ms,
                "calibration_confidence": float(calibration_confidence),
            },
        )
