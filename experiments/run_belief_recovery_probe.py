import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photonic_sim import (  # noqa: E402
    ActionExecutorConfig,
    AgentEnv,
    CalibrationBootstrap,
    CalibrationState,
    MRRArrayPlant,
    MRRPlantConfig,
    OSAInstrument,
    OSAInstrumentConfig,
    PDInstrument,
    PDInstrumentConfig,
    RecoveryTrigger,
    SimulationRuntime,
    SimpleBeliefStateEstimator,
    TaskSpec,
    build_comb_wavelengths,
    simulate_target_resonances,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Probe belief-state updates and recovery triggers.")
    parser.add_argument("--num-rings", type=int, default=3)
    parser.add_argument("--settle-ms", type=float, default=20.0)
    parser.add_argument("--bootstrap-dir", type=str, default="experiments/outputs/baseline_20260311")
    parser.add_argument("--output-dir", type=str, default="experiments/outputs/belief_recovery_probe")
    return parser.parse_args()


def build_runtime_factory(num_rings: int):
    def factory():
        comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=num_rings)
        plant = MRRArrayPlant(
            num_rings=num_rings,
            comb_wavelengths_nm=comb,
            config=MRRPlantConfig(
                thermal_tau_ms=8.0,
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
                    full_scale_current_ma=0.001,
                    frame_period_ms=0.0,
                    noise_sigma=0.0,
                ),
                rng=np.random.default_rng(1),
            ),
            osa_instrument=OSAInstrument(
                OSAInstrumentConfig(
                    step_pm=5.0,
                    span_nm=1.2,
                    frame_period_ms=20.0,
                    amplitude_noise_sigma=0.0,
                ),
                rng=np.random.default_rng(2),
            ),
        )

    return factory


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_probe_row(rows, step_name, action_type, observation, belief_state, recovery_decision):
    rows.append(
        {
            "step_name": step_name,
            "action_type": action_type,
            "time_ms": observation.time_ms,
            "max_uncertainty_pm": float(np.max(belief_state.resonance_uncertainty_pm)),
            "mean_uncertainty_pm": float(np.mean(belief_state.resonance_uncertainty_pm)),
            "max_innovation_pm": 0.0
            if belief_state.innovation_pm is None
            else float(np.max(np.abs(belief_state.innovation_pm))),
            "min_identity_confidence": float(np.min(belief_state.identity_confidence)),
            "calibration_confidence": belief_state.calibration_confidence,
            "latest_osa_quality": None
            if observation.latest_osa_frame is None
            else observation.latest_osa_frame.quality_flag,
            "latest_pd_quality": None
            if observation.latest_pd_frame is None
            else observation.latest_pd_frame.quality_flag,
            "recovery_should_trigger": recovery_decision.should_recover,
            "recovery_action": recovery_decision.suggested_action,
            "recovery_stale_measurement": recovery_decision.stale_measurement,
            "recovery_saturation_detected": recovery_decision.saturation_detected,
            "recovery_belief_divergence": recovery_decision.belief_divergence,
            "recovery_collision_suspected": recovery_decision.collision_suspected,
            "recovery_calibration_confidence_low": recovery_decision.calibration_confidence_low,
        }
    )


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_factory = build_runtime_factory(args.num_rings)
    target_voltages_v = np.linspace(0.9, 1.7, args.num_rings)
    initial_voltages_v = np.clip(target_voltages_v + np.linspace(0.6, -0.3, args.num_rings), 0.0, 5.0)
    target_resonances_nm = simulate_target_resonances(
        runtime_factory=runtime_factory,
        target_voltages_v=target_voltages_v,
        settle_ms=args.settle_ms,
    )

    bootstrap_dir = ROOT / args.bootstrap_dir
    calibration_state = None
    if bootstrap_dir.exists():
        calibration_result = CalibrationBootstrap.fit_from_experiment_dir(bootstrap_dir)
        calibration_state = CalibrationState.from_bootstrap_result(
            calibration_result,
            num_rings=args.num_rings,
            confidence=0.85,
        )
    else:
        calibration_state = CalibrationState(
            version="default",
            source="runtime-default",
            tuning_efficiency_nm_per_mw=0.015,
            thermal_t63_ms=8.0,
            thermal_t95_ms=24.0,
            crosstalk_matrix=np.eye(args.num_rings),
            recommended_pd_config={},
            recommended_osa_config={"step_pm": 5.0, "span_nm": 1.2},
            confidence=0.7,
        )

    env = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(
            target_resonances_nm=target_resonances_nm,
            tolerance_pm=1.0,
            max_episode_time_ms=120.0,
            max_control_steps=20,
            success_hold_steps=1,
            action_budget=30.0,
            observation_budget=30.0,
            allowed_instruments=("OSA", "PD"),
        ),
    )
    observation = env.reset(initial_voltages_v=initial_voltages_v, settle_ms=args.settle_ms)

    estimator = SimpleBeliefStateEstimator()
    recovery_trigger = RecoveryTrigger()
    belief_state = estimator.initialize(observation, calibration_state)
    recovery_decision = recovery_trigger.evaluate(observation, belief_state, calibration_state)
    belief_rows = []
    append_probe_row(belief_rows, "reset", "reset", observation, belief_state, recovery_decision)

    sequence = [
        (
            "fresh_osa",
            {
                "type": "read_osa",
                "center_nm": float(np.mean(target_resonances_nm)),
                "span_nm": 1.2,
            },
        ),
        ("wait_before_stale_osa", {"type": "wait", "dt_ms": 10.0}),
        (
            "stale_osa",
            {
                "type": "read_osa",
                "center_nm": float(np.mean(target_resonances_nm)),
                "span_nm": 1.2,
            },
        ),
        (
            "pd_saturation_probe",
            {
                "type": "read_pd",
                "input_powers_mw": np.full(args.num_rings, 100.0),
            },
        ),
    ]

    for step_name, action in sequence:
        step_result = env.step(action)
        belief_state = estimator.update(belief_state, step_result.observation, calibration_state)
        recovery_decision = recovery_trigger.evaluate(step_result.observation, belief_state, calibration_state)
        append_probe_row(
            belief_rows,
            step_name,
            action["type"],
            step_result.observation,
            belief_state,
            recovery_decision,
        )

    write_csv(output_dir / "belief_probe.csv", belief_rows)
    print(f"saved: {output_dir / 'belief_probe.csv'}")


if __name__ == "__main__":
    main()
