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
