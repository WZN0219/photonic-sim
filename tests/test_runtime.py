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


def test_plant_snapshot_restore_preserves_future_evolution():
    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=3)
    plant = MRRArrayPlant(
        num_rings=3,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(thermal_tau_ms=7.0, drift_sigma_nm_per_s=0.003),
        rng=np.random.default_rng(11),
        action_config=ActionExecutorConfig(slew_rate_v_per_ms=2.0),
    )
    plant.issue_command(channel=1, target_voltage_v=2.5)
    plant.step(3.0)
    snapshot = plant.snapshot()

    restored = MRRArrayPlant(
        num_rings=3,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(thermal_tau_ms=7.0, drift_sigma_nm_per_s=0.003),
        rng=np.random.default_rng(0),
        action_config=ActionExecutorConfig(slew_rate_v_per_ms=2.0),
    )
    restored.restore(snapshot)

    live_future = plant.step(5.0)
    restored_future = restored.step(5.0)

    assert np.allclose(live_future.actuator_voltages_v, restored_future.actuator_voltages_v)
    assert np.allclose(live_future.thermal_powers_mw, restored_future.thermal_powers_mw)
    assert np.allclose(live_future.drift_nm, restored_future.drift_nm)
    assert np.allclose(live_future.effective_resonances_nm, restored_future.effective_resonances_nm)


def test_runtime_fork_preserves_measurement_cache_and_rng():
    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=3)
    runtime = SimulationRuntime(
        plant=MRRArrayPlant(
            num_rings=3,
            comb_wavelengths_nm=comb,
            config=MRRPlantConfig(drift_sigma_nm_per_s=0.0),
            rng=np.random.default_rng(12),
        ),
        pd_instrument=PDInstrument(
            PDInstrumentConfig(frame_period_ms=2.0, noise_sigma=0.0),
            rng=np.random.default_rng(13),
        ),
        osa_instrument=OSAInstrument(
            OSAInstrumentConfig(frame_period_ms=5.0, amplitude_noise_sigma=0.0),
            rng=np.random.default_rng(14),
        ),
    )

    runtime.read_pd()
    runtime.read_osa()
    clone = runtime.fork()

    runtime.step(1.0)
    clone.step(1.0)
    pd_live = runtime.read_pd()
    pd_clone = clone.read_pd()
    assert pd_live.quality_flag == pd_clone.quality_flag == "stale"
    assert pd_live.metadata["source_timestamp_ms"] == pd_clone.metadata["source_timestamp_ms"]

    runtime.step(5.0)
    clone.step(5.0)
    osa_live = runtime.read_osa()
    osa_clone = clone.read_osa()
    assert osa_live.quality_flag == osa_clone.quality_flag == "fresh"
    assert np.allclose(osa_live.payload["wavelengths_nm"], osa_clone.payload["wavelengths_nm"])
    assert np.allclose(osa_live.payload["spectrum_dbm"], osa_clone.payload["spectrum_dbm"])


def test_plant_restore_rebuilds_geometry_from_snapshot():
    comb1 = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=3)
    comb2 = build_comb_wavelengths(center_nm=1560.0, fsr_nm=0.73, num_lines=3)
    source = MRRArrayPlant(
        num_rings=3,
        comb_wavelengths_nm=comb1,
        config=MRRPlantConfig(q_factor=5000.0, crosstalk_alpha=0.08, drift_sigma_nm_per_s=0.0),
        rng=np.random.default_rng(21),
        action_config=ActionExecutorConfig(slew_rate_v_per_ms=2.0),
    )
    source.issue_command(channel=1, target_voltage_v=2.5)
    source.step(4.0)
    snapshot = source.snapshot()

    recipient = MRRArrayPlant(
        num_rings=3,
        comb_wavelengths_nm=comb2,
        config=MRRPlantConfig(q_factor=10000.0, crosstalk_alpha=0.2, drift_sigma_nm_per_s=0.0),
        rng=np.random.default_rng(22),
        action_config=ActionExecutorConfig(slew_rate_v_per_ms=0.5),
    )
    recipient.restore(snapshot)

    assert np.allclose(recipient.comb_wavelengths_nm, source.comb_wavelengths_nm)
    assert np.allclose(recipient.base_resonances_nm, source.base_resonances_nm)
    assert np.allclose(recipient.bandwidth_nm, source.bandwidth_nm)
    assert np.allclose(recipient.hwhm_nm, source.hwhm_nm)
    assert np.allclose(recipient.crosstalk_matrix, source.crosstalk_matrix)
    assert np.allclose(recipient.latent_state().effective_resonances_nm, source.latent_state().effective_resonances_nm)


def test_stale_frames_preserve_measurement_metadata():
    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=3)
    plant = MRRArrayPlant(
        num_rings=3,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(drift_sigma_nm_per_s=0.0),
        rng=np.random.default_rng(23),
    )
    plant.set_global_temp_shift_nm(plant.config.fsr_nm / 2.0)
    pd = PDInstrument(
        PDInstrumentConfig(adc_bits=4, full_scale_current_ma=0.05, frame_period_ms=2.0, noise_sigma=0.0),
        rng=np.random.default_rng(24),
    )
    fresh_pd = pd.sample(plant, input_powers_mw=np.full(3, 10.0))
    stale_pd = pd.sample(plant, input_powers_mw=np.full(3, 10.0))
    assert fresh_pd.metadata["saturated"] is True
    assert stale_pd.metadata["saturated"] is True
    assert stale_pd.metadata["adc_bits"] == fresh_pd.metadata["adc_bits"]
    assert stale_pd.metadata["adc_lsb_ma"] == fresh_pd.metadata["adc_lsb_ma"]

    osa = OSAInstrument(
        OSAInstrumentConfig(step_pm=5.0, span_nm=0.6, frame_period_ms=5.0, amplitude_noise_sigma=0.0),
        rng=np.random.default_rng(25),
    )
    fresh_osa = osa.sample(plant, center_nm=1550.0, span_nm=0.6)
    stale_osa = osa.sample(plant, center_nm=1550.0, span_nm=0.6)
    assert stale_osa.metadata["center_nm"] == fresh_osa.metadata["center_nm"]
    assert stale_osa.metadata["span_nm"] == fresh_osa.metadata["span_nm"]
    assert stale_osa.metadata["step_pm"] == fresh_osa.metadata["step_pm"]
