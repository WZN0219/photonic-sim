import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photonic_sim import (  # noqa: E402
    ActionExecutorConfig,
    MRRArrayPlant,
    MRRPlantConfig,
    OSAInstrument,
    OSAInstrumentConfig,
    PDInstrument,
    PDInstrumentConfig,
    SafetyGuardConfig,
    SimulationRuntime,
    build_comb_wavelengths,
)


def test_actuator_and_thermal_state_require_time_to_take_effect():
    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=3)
    plant = MRRArrayPlant(
        num_rings=3,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(thermal_tau_ms=10.0, drift_sigma_nm_per_s=0.0),
        rng=np.random.default_rng(0),
        action_config=ActionExecutorConfig(slew_rate_v_per_ms=10.0),
    )

    ack = plant.issue_command(channel=1, target_voltage_v=4.0)
    assert ack.status == "accepted"
    initial = plant.latent_state()
    assert initial.actuator_voltages_v[1] == 0.0
    assert initial.thermal_powers_mw[1] == 0.0

    mid = plant.step(1.0)
    assert mid.actuator_voltages_v[1] == 4.0
    assert 0.0 < mid.thermal_powers_mw[1] < mid.command_powers_mw[1]

    late = plant.step(50.0)
    assert late.actuator_voltages_v[1] == mid.actuator_voltages_v[1]
    assert late.thermal_powers_mw[1] > mid.thermal_powers_mw[1]
    assert late.effective_resonances_nm[1] > initial.effective_resonances_nm[1]


def test_executor_clamps_and_safety_reports():
    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=2)
    plant = MRRArrayPlant(
        num_rings=2,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(thermal_tau_ms=1.0, drift_sigma_nm_per_s=0.0),
        rng=np.random.default_rng(10),
        action_config=ActionExecutorConfig(max_voltage_v=3.0, slew_rate_v_per_ms=10.0),
        safety_config=SafetyGuardConfig(max_self_shift_nm=0.02, max_total_shift_nm=0.015),
    )

    ack = plant.issue_command(channel=0, target_voltage_v=6.0)
    assert ack.status == "clamped"
    assert ack.target_voltage_v == 3.0

    state = plant.step(10.0)
    assert state.self_shift_clamped_mask[0]
    assert np.any(state.total_shift_warning_mask)


def test_pd_uses_fixed_full_scale_and_saturation():
    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=3)
    plant = MRRArrayPlant(
        num_rings=3,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(drift_sigma_nm_per_s=0.0),
        rng=np.random.default_rng(1),
    )
    plant.set_global_temp_shift_nm(plant.config.fsr_nm / 2.0)
    pd = PDInstrument(
        PDInstrumentConfig(
            adc_bits=4,
            full_scale_current_ma=0.05,
            frame_period_ms=0.0,
            noise_sigma=0.0,
        ),
        rng=np.random.default_rng(2),
    )

    low = pd.sample(plant, input_powers_mw=np.full(3, 0.1))
    high = pd.sample(plant, input_powers_mw=np.full(3, 10.0))

    assert low.metadata["adc_lsb_ma"] == high.metadata["adc_lsb_ma"]
    assert high.metadata["saturated"] is True
    assert np.max(high.payload["quantized_currents_ma"]) <= 0.05 + 1e-12


def test_runtime_returns_timestamped_fresh_and_stale_frames():
    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=3)
    plant = MRRArrayPlant(
        num_rings=3,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(drift_sigma_nm_per_s=0.0),
        rng=np.random.default_rng(3),
    )
    runtime = SimulationRuntime(
        plant=plant,
        pd_instrument=PDInstrument(
            PDInstrumentConfig(frame_period_ms=2.0, noise_sigma=0.0),
            rng=np.random.default_rng(4),
        ),
        osa_instrument=OSAInstrument(
            OSAInstrumentConfig(frame_period_ms=5.0, amplitude_noise_sigma=0.0),
            rng=np.random.default_rng(5),
        ),
    )

    pd_frame_1 = runtime.read_pd()
    assert pd_frame_1.instrument_type == "PD"
    assert pd_frame_1.quality_flag == "fresh"
    assert pd_frame_1.timestamp_ms == 0.0

    runtime.step(1.0)
    pd_frame_2 = runtime.read_pd()
    assert pd_frame_2.quality_flag == "stale"
    assert pd_frame_2.metadata["source_timestamp_ms"] == 0.0

    runtime.step(2.0)
    pd_frame_3 = runtime.read_pd()
    assert pd_frame_3.quality_flag == "fresh"
    assert pd_frame_3.timestamp_ms == 3.0

    osa_frame = runtime.read_osa()
    assert osa_frame.instrument_type == "OSA"
    assert "wavelengths_nm" in osa_frame.payload
    assert osa_frame.calib_version == "osa-v1"
