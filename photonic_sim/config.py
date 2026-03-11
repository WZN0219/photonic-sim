from dataclasses import dataclass


@dataclass(frozen=True)
class MRRPlantConfig:
    base_resonance_nm: float = 1550.0
    fsr_nm: float = 0.73
    q_factor: float = 5000.0
    extinction_ratio_db: float = 25.0
    heater_resistance_ohm: float = 1200.0
    tuning_efficiency_nm_per_mw: float = 0.015
    crosstalk_alpha: float = 0.08
    crosstalk_decay_length: float = 2.0
    thermal_tau_ms: float = 5.0
    drift_sigma_nm_per_s: float = 0.002
    initial_global_temp_shift_nm: float = 0.0


@dataclass(frozen=True)
class ActionExecutorConfig:
    max_voltage_v: float = 5.0
    slew_rate_v_per_ms: float = 1.0


@dataclass(frozen=True)
class SafetyGuardConfig:
    max_self_shift_nm: float = 0.25
    max_total_shift_nm: float = 0.35


@dataclass(frozen=True)
class PDInstrumentConfig:
    responsivity_aw: float = 0.8
    dark_current_na: float = 1.0
    noise_sigma: float = 0.002
    adc_bits: int = 10
    full_scale_current_ma: float = 1.0
    frame_period_ms: float = 1.0
    calib_version: str = "pd-v1"


@dataclass(frozen=True)
class OSAInstrumentConfig:
    step_pm: float = 10.0
    span_nm: float = 2.0
    frame_period_ms: float = 100.0
    noise_floor_mw: float = 1e-9
    amplitude_noise_sigma: float = 0.001
    calib_version: str = "osa-v1"
