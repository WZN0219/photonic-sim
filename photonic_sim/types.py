from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LatentPlantState:
    time_ms: float
    actuator_voltages_v: np.ndarray
    target_voltages_v: np.ndarray
    command_powers_mw: np.ndarray
    thermal_powers_mw: np.ndarray
    ideal_shifts_nm: np.ndarray
    drift_nm: np.ndarray
    effective_resonances_nm: np.ndarray
    global_temp_shift_nm: float
    self_shift_clamped_mask: np.ndarray
    total_shift_warning_mask: np.ndarray


@dataclass
class MeasurementFrame:
    instrument_type: str
    timestamp_ms: float
    calib_version: str
    quality_flag: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionAck:
    channel: int
    requested_voltage_v: float
    target_voltage_v: float
    issued_at_ms: float
    status: str
    message: str


@dataclass
class PlantSnapshot:
    num_rings: int
    comb_wavelengths_nm: np.ndarray
    config: Any
    action_config: Any
    safety_config: Any
    rng_state: dict[str, Any]
    time_ms: float
    target_voltages_v: np.ndarray
    actuator_voltages_v: np.ndarray
    command_powers_mw: np.ndarray
    thermal_powers_mw: np.ndarray
    drift_nm: np.ndarray
    global_temp_shift_nm: float
    ideal_shifts_nm: np.ndarray
    effective_resonances_nm: np.ndarray
    self_shift_clamped_mask: np.ndarray
    total_shift_warning_mask: np.ndarray


@dataclass
class RuntimeSnapshot:
    plant_snapshot: PlantSnapshot
    pd_snapshot: Any
    osa_snapshot: Any
    action_log: list[dict[str, Any]]
    measurement_log: list[dict[str, Any]]


@dataclass
class BudgetAccountantSnapshot:
    action_budget_used: float
    observation_budget_used: float
    num_voltage_commands: int
    num_wait_actions: int
    num_pd_reads: int
    num_osa_reads: int
    num_clamped_actions: int
    num_saturated_measurements: int
    num_safety_warnings: int
    num_safety_warning_active_steps: int
    previous_warning_mask: np.ndarray | None


@dataclass
class AgentEnvSnapshot:
    task_spec: Any
    runtime_snapshot: RuntimeSnapshot
    accountant_snapshot: BudgetAccountantSnapshot
    episode_log: list[dict[str, Any]]
    episode_start_ms: float
    step_index: int
    success_streak: int
    previous_mean_abs_error_pm: float
    done: bool
    done_reason: str
    last_action: dict[str, Any] | None
    last_ack: ActionAck | None
    latest_pd_frame: MeasurementFrame | None
    latest_osa_frame: MeasurementFrame | None
