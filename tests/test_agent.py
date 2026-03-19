import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photonic_sim import (  # noqa: E402
    ActionExecutorConfig,
    AgentEnv,
    BootstrapRetuningController,
    BootstrapRetuningControllerConfig,
    BudgetConfig,
    MRRArrayPlant,
    MRRPlantConfig,
    OSAInstrument,
    OSAInstrumentConfig,
    PDInstrument,
    PDInstrumentConfig,
    SafetyGuardConfig,
    SimulationRuntime,
    TaskSpec,
    build_comb_wavelengths,
    simulate_target_resonances,
)


def _build_runtime_factory(
    num_rings: int,
    thermal_tau_ms: float = 8.0,
    osa_frame_period_ms: float = 0.0,
    safety_config: Optional[SafetyGuardConfig] = None,
):
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
            safety_config=safety_config,
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
                    span_nm=0.6,
                    frame_period_ms=osa_frame_period_ms,
                    amplitude_noise_sigma=0.0,
                ),
                rng=np.random.default_rng(2),
            ),
        )

    return factory


def test_agent_env_tracks_budget_and_stops_when_finished():
    runtime_factory = _build_runtime_factory(num_rings=1)
    target_resonances_nm = simulate_target_resonances(
        runtime_factory=runtime_factory,
        target_voltages_v=np.array([1.5]),
        settle_ms=12.0,
    )
    env = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(
            target_resonances_nm=target_resonances_nm,
            tolerance_pm=5.0,
            max_episode_time_ms=100.0,
            max_control_steps=10,
            success_hold_steps=1,
            action_budget=10.0,
            observation_budget=10.0,
            allowed_instruments=("OSA", "PD"),
        ),
        budget_config=BudgetConfig(
            voltage_action_cost=1.0,
            wait_cost_per_ms=0.1,
            pd_read_cost=2.0,
            osa_read_cost=3.0,
        ),
    )
    observation = env.reset(initial_voltages_v=np.array([1.5]), settle_ms=12.0)
    assert observation.commanded_voltages_v[0] == 1.5

    result = env.step({"type": "read_osa", "center_nm": float(target_resonances_nm[0]), "span_nm": 0.3})
    assert result.observation.budget_state.num_osa_reads == 1
    assert result.observation.budget_state.observation_budget_used == 3.0

    finish = env.step({"type": "finish"})
    assert finish.done is True
    assert finish.info["success"] is True
    assert finish.info["done_reason"] in ("converged", "agent_finish_success")


def test_bootstrap_retuning_controller_reduces_single_ring_error():
    runtime_factory = _build_runtime_factory(num_rings=1)
    target_resonances_nm = simulate_target_resonances(
        runtime_factory=runtime_factory,
        target_voltages_v=np.array([2.0]),
        settle_ms=15.0,
    )
    env = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(
            target_resonances_nm=target_resonances_nm,
            tolerance_pm=8.0,
            max_episode_time_ms=150.0,
            max_control_steps=20,
            success_hold_steps=1,
            action_budget=30.0,
            observation_budget=30.0,
            allowed_instruments=("OSA",),
        ),
        budget_config=BudgetConfig(
            voltage_action_cost=1.0,
            wait_cost_per_ms=0.05,
            osa_read_cost=2.0,
        ),
    )
    env.reset(initial_voltages_v=np.array([4.0]), settle_ms=15.0)
    initial_error_pm = env.episode_summary()["final_max_abs_error_pm"]

    controller = BootstrapRetuningController(
        config=BootstrapRetuningControllerConfig(
            osa_span_nm=0.4,
            settle_ms=10.0,
            correction_gain=0.9,
            max_rounds=3,
        ),
    )
    summary = controller.run_episode(env)

    assert summary["final_max_abs_error_pm"] < initial_error_pm
    assert summary["num_osa_reads"] >= 1


def test_safety_warning_counts_new_events_without_repenalizing_persistent_warning():
    runtime_factory = _build_runtime_factory(
        num_rings=1,
        safety_config=SafetyGuardConfig(max_self_shift_nm=0.25, max_total_shift_nm=0.1),
    )
    target_resonances_nm = simulate_target_resonances(
        runtime_factory=runtime_factory,
        target_voltages_v=np.array([0.0]),
        settle_ms=0.0,
    )
    env = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(
            target_resonances_nm=target_resonances_nm,
            tolerance_pm=5.0,
            max_episode_time_ms=100.0,
            max_control_steps=10,
            success_hold_steps=2,
            action_budget=20.0,
            observation_budget=20.0,
            allowed_instruments=("OSA",),
        ),
        budget_config=BudgetConfig(
            voltage_action_cost=1.0,
            wait_cost_per_ms=0.0,
            osa_read_cost=2.0,
            safety_warning_cost=0.5,
        ),
    )
    env.reset(initial_voltages_v=np.array([0.0]), settle_ms=0.0)
    env.step({"type": "set_voltage", "channel": 0, "voltage_v": 5.0})

    warning_wait = env.step({"type": "wait", "dt_ms": 40.0})
    assert warning_wait.observation.budget_state.num_safety_warnings == 1
    assert warning_wait.observation.budget_state.num_safety_warning_active_steps == 1

    warning_read = env.step({"type": "read_osa", "center_nm": float(target_resonances_nm[0]), "span_nm": 0.6})
    assert warning_read.observation.budget_state.num_safety_warnings == 1
    assert warning_read.observation.budget_state.num_safety_warning_active_steps == 2
    assert warning_read.reward == -2.0


def test_reset_per_ring_static_shift_matches_manual_shift_injection():
    runtime_factory = _build_runtime_factory(num_rings=2)
    target_resonances_nm = simulate_target_resonances(
        runtime_factory=runtime_factory,
        target_voltages_v=np.array([1.0, 1.5]),
        settle_ms=10.0,
    )
    static_shift = np.array([0.012, -0.008], dtype=float)

    env_new = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(target_resonances_nm=target_resonances_nm, allowed_instruments=("OSA",)),
    )
    env_old = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(target_resonances_nm=target_resonances_nm, allowed_instruments=("OSA",)),
    )

    env_new.reset(initial_voltages_v=np.array([1.0, 1.5]), settle_ms=10.0, per_ring_static_shift_nm=static_shift)
    env_old.reset(initial_voltages_v=np.array([1.0, 1.5]), settle_ms=10.0)
    env_old.runtime.plant.drift_nm = static_shift.copy()
    env_old.runtime.plant._recompute_latent_state()

    state_new = env_new.runtime.plant.latent_state()
    state_old = env_old.runtime.plant.latent_state()
    assert np.allclose(state_new.drift_nm, state_old.drift_nm)
    assert np.allclose(state_new.effective_resonances_nm, state_old.effective_resonances_nm)


def test_agent_env_fork_preserves_rollout_exactly():
    runtime_factory = _build_runtime_factory(num_rings=1, osa_frame_period_ms=5.0)
    target_resonances_nm = simulate_target_resonances(
        runtime_factory=runtime_factory,
        target_voltages_v=np.array([1.6]),
        settle_ms=12.0,
    )
    env = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(
            target_resonances_nm=target_resonances_nm,
            tolerance_pm=5.0,
            max_episode_time_ms=100.0,
            max_control_steps=10,
            success_hold_steps=1,
            action_budget=20.0,
            observation_budget=20.0,
            allowed_instruments=("OSA",),
        ),
        budget_config=BudgetConfig(
            voltage_action_cost=1.0,
            wait_cost_per_ms=0.05,
            osa_read_cost=2.0,
        ),
    )
    env.reset(initial_voltages_v=np.array([3.5]), settle_ms=12.0, per_ring_static_shift_nm=np.array([0.01]))
    env.step({"type": "set_voltage", "channel": 0, "voltage_v": 2.5})
    clone = env.fork()

    actions = [
        {"type": "wait", "dt_ms": 10.0},
        {"type": "read_osa", "center_nm": float(target_resonances_nm[0]), "span_nm": 0.4},
        {"type": "finish"},
    ]
    for action in actions:
        live = env.step(action)
        replay = clone.step(action)
        assert live.done == replay.done
        assert live.info["done_reason"] == replay.info["done_reason"]
        assert np.isclose(live.reward, replay.reward)
        assert np.isclose(
            live.observation.budget_state.observation_budget_used,
            replay.observation.budget_state.observation_budget_used,
        )
        assert np.allclose(
            env.runtime.plant.latent_state().effective_resonances_nm,
            clone.runtime.plant.latent_state().effective_resonances_nm,
        )


def test_agent_env_restore_replaces_task_spec_from_snapshot():
    runtime_factory = _build_runtime_factory(num_rings=1)
    source_target = simulate_target_resonances(
        runtime_factory=runtime_factory,
        target_voltages_v=np.array([1.5]),
        settle_ms=0.0,
    )
    env = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(target_resonances_nm=source_target, allowed_instruments=("OSA",)),
        budget_config=BudgetConfig(),
    )
    env.reset(initial_voltages_v=np.array([0.0]), settle_ms=0.0)
    snapshot = env.snapshot()

    wrong_target = np.array([1600.0], dtype=float)
    restored = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(target_resonances_nm=wrong_target, allowed_instruments=("OSA",)),
        budget_config=BudgetConfig(),
    )
    restored.restore(snapshot)

    assert np.allclose(restored.task_spec.target_resonances_nm, source_target)
    assert restored.episode_summary()["final_max_abs_error_pm"] == env.episode_summary()["final_max_abs_error_pm"]
