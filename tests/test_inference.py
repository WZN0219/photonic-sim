import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photonic_sim import (  # noqa: E402
    ActionExecutorConfig,
    AgentEnv,
    BeliefState,
    CalibrationState,
    RecoveryConfig,
    RecoveryTrigger,
    SimpleBeliefStateEstimator,
    SimpleBeliefStateEstimatorConfig,
    TaskSpec,
    build_comb_wavelengths,
    simulate_target_resonances,
)
from photonic_sim.calibration import (  # noqa: E402
    CalibrationBootstrapResult,
    CrosstalkCalibration,
    DriftCalibration,
    ObservationCalibration,
    StepResponseCalibration,
)
from photonic_sim.instruments import OSAInstrument, OSAInstrumentConfig, PDInstrument, PDInstrumentConfig  # noqa: E402
from photonic_sim.plant import MRRArrayPlant  # noqa: E402
from photonic_sim.config import MRRPlantConfig  # noqa: E402
from photonic_sim.runtime import SimulationRuntime  # noqa: E402


def _build_runtime_factory(num_rings: int, thermal_tau_ms: float = 8.0, osa_frame_period_ms: float = 20.0):
    def factory():
        comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=num_rings)
        plant = MRRArrayPlant(
            num_rings=num_rings,
            comb_wavelengths_nm=comb,
            config=MRRPlantConfig(
                thermal_tau_ms=thermal_tau_ms,
                drift_sigma_nm_per_s=0.0,
            ),
            rng=np.random.default_rng(0),
            action_config=ActionExecutorConfig(
                max_voltage_v=5.0,
                slew_rate_v_per_ms=10.0,
            ),
        )
        return SimulationRuntime(
            plant=plant,
            pd_instrument=PDInstrument(
                PDInstrumentConfig(
                    adc_bits=8,
                    full_scale_current_ma=0.2,
                    frame_period_ms=0.0,
                    noise_sigma=0.0,
                ),
                rng=np.random.default_rng(1),
            ),
            osa_instrument=OSAInstrument(
                OSAInstrumentConfig(
                    step_pm=5.0,
                    span_nm=1.8,
                    frame_period_ms=osa_frame_period_ms,
                    amplitude_noise_sigma=0.0,
                ),
                rng=np.random.default_rng(2),
            ),
        )

    return factory


def test_calibration_state_builds_from_bootstrap_result_and_rescales_matrix():
    bootstrap = CalibrationBootstrapResult(
        source_dir="E:/tmp/mock_exp",
        step_response=StepResponseCalibration(
            final_shift_nm=0.2,
            final_effective_resonance_nm=1550.2,
            final_command_power_mw=12.0,
            final_thermal_power_mw=10.0,
            estimated_tuning_efficiency_nm_per_mw=0.02,
            t63_ms=8.0,
            t95_ms=24.0,
        ),
        crosstalk=CrosstalkCalibration(
            drive_channel=1,
            center_shift_nm=0.2,
            relative_profile_by_offset={-2: 0.04, -1: 0.1, 0: 1.0, 1: 0.1, 2: 0.04},
            estimated_crosstalk_matrix=[
                [1.0, 0.1, 0.04],
                [0.1, 1.0, 0.1],
                [0.04, 0.1, 1.0],
            ],
        ),
        observation=ObservationCalibration(
            recommended_pd_config={"adc_bits": 8},
            recommended_osa_config={"step_pm": 5.0},
            pd_summary={},
            osa_summary={},
        ),
        drift=DriftCalibration(
            duration_ms=200.0,
            latent_rows=200,
            pd_rows=40,
            osa_rows=810,
            pd_frame_period_ms=5.0,
            osa_frame_period_ms=20.0,
            resonance_span_pm_by_ring={0: 10.0},
        ),
    )

    state = CalibrationState.from_bootstrap_result(bootstrap, num_rings=5, confidence=0.8, timestamp_ms=12.0)
    assert state.version == "bootstrap:mock_exp"
    assert state.crosstalk_matrix.shape == (5, 5)
    assert round(state.crosstalk_matrix[0, 1], 6) == 0.1
    assert round(state.crosstalk_matrix[0, 2], 6) == 0.04
    assert round(state.confidence, 6) == 0.8

    resized = state.for_num_rings(3)
    assert resized.crosstalk_matrix.shape == (3, 3)
    assert round(resized.crosstalk_matrix[0, 1], 6) == 0.1


def test_simple_belief_estimator_updates_from_fresh_osa_and_then_grows_uncertainty():
    runtime_factory = _build_runtime_factory(num_rings=3, osa_frame_period_ms=20.0)
    target_resonances_nm = simulate_target_resonances(
        runtime_factory=runtime_factory,
        target_voltages_v=np.array([0.8, 1.2, 1.6]),
        settle_ms=20.0,
    )
    env = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(
            target_resonances_nm=target_resonances_nm,
            tolerance_pm=10.0,
            max_episode_time_ms=100.0,
            max_control_steps=20,
            success_hold_steps=1,
            action_budget=30.0,
            observation_budget=30.0,
            allowed_instruments=("OSA",),
        ),
    )
    env.reset(initial_voltages_v=np.array([1.4, 1.4, 1.4]), settle_ms=20.0)
    calibration_state = CalibrationState(
        version="unit",
        source="unit-test",
        tuning_efficiency_nm_per_mw=0.015,
        thermal_t63_ms=8.0,
        thermal_t95_ms=24.0,
        crosstalk_matrix=np.eye(3),
        recommended_pd_config={},
        recommended_osa_config={"step_pm": 5.0},
        confidence=0.9,
    )
    estimator = SimpleBeliefStateEstimator(
        SimpleBeliefStateEstimatorConfig(
            osa_local_window_nm=0.5,
            initial_uncertainty_pm=100.0,
            fresh_osa_uncertainty_pm=5.0,
            uncertainty_growth_pm_per_ms=0.2,
        )
    )

    belief_0 = estimator.initialize(env.observation(), calibration_state)
    assert round(float(np.max(belief_0.resonance_uncertainty_pm)), 6) == 100.0

    osa_step = env.step(
        {
            "type": "read_osa",
            "center_nm": float(np.mean(target_resonances_nm)),
            "span_nm": 1.8,
        }
    )
    belief_1 = estimator.update(belief_0, osa_step.observation, calibration_state)
    actual_resonances_nm = env.runtime.plant.latent_state().effective_resonances_nm
    assert float(np.max(np.abs(belief_1.resonance_estimates_nm - actual_resonances_nm)) * 1000.0) <= 5.0
    assert round(float(np.max(belief_1.resonance_uncertainty_pm)), 6) <= 7.0

    wait_step = env.step({"type": "wait", "dt_ms": 10.0})
    belief_2 = estimator.update(belief_1, wait_step.observation, calibration_state)
    assert np.all(belief_2.resonance_uncertainty_pm > belief_1.resonance_uncertainty_pm)


def test_recovery_trigger_flags_stale_divergence_collision_and_low_confidence():
    runtime_factory = _build_runtime_factory(num_rings=2, osa_frame_period_ms=20.0)
    target_resonances_nm = simulate_target_resonances(
        runtime_factory=runtime_factory,
        target_voltages_v=np.array([1.0, 1.4]),
        settle_ms=15.0,
    )
    env = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(
            target_resonances_nm=target_resonances_nm,
            tolerance_pm=1.0,
            max_episode_time_ms=100.0,
            max_control_steps=20,
            success_hold_steps=1,
            action_budget=30.0,
            observation_budget=30.0,
            allowed_instruments=("OSA",),
        ),
    )
    calibration_state = CalibrationState(
        version="unit",
        source="unit-test",
        tuning_efficiency_nm_per_mw=0.015,
        thermal_t63_ms=8.0,
        thermal_t95_ms=24.0,
        crosstalk_matrix=np.eye(2),
        recommended_pd_config={},
        recommended_osa_config={},
        confidence=0.2,
    )

    env.reset(initial_voltages_v=np.array([1.0, 1.0]), settle_ms=15.0)
    env.step({"type": "read_osa", "center_nm": float(np.mean(target_resonances_nm)), "span_nm": 1.2})
    env.step({"type": "wait", "dt_ms": 10.0})
    stale_step = env.step({"type": "read_osa", "center_nm": float(np.mean(target_resonances_nm)), "span_nm": 1.2})
    assert stale_step.observation.latest_osa_frame.quality_flag == "stale"

    belief_state = BeliefState(
        time_ms=stale_step.observation.time_ms,
        resonance_estimates_nm=np.array([1550.000, 1550.020]),
        resonance_uncertainty_pm=np.array([20.0, 20.0]),
        identity_confidence=np.array([0.4, 0.4]),
        calibration_confidence=0.2,
        innovation_pm=np.array([35.0, 5.0]),
    )
    decision = RecoveryTrigger(
        RecoveryConfig(
            stale_age_threshold_ms=5.0,
            belief_divergence_threshold_pm=25.0,
            collision_margin_pm=30.0,
            calibration_confidence_threshold=0.3,
        )
    ).evaluate(stale_step.observation, belief_state, calibration_state)

    assert decision.should_recover is True
    assert decision.stale_measurement is True
    assert decision.belief_divergence is True
    assert decision.collision_suspected is True
    assert decision.calibration_confidence_low is True
    assert decision.suggested_action == "rebootstrap"
