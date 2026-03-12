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
    BootstrapRetuningController,
    BootstrapRetuningControllerConfig,
    BudgetConfig,
    CalibrationBootstrap,
    CalibrationState,
    MRRArrayPlant,
    MRRPlantConfig,
    OSAInstrument,
    OSAInstrumentConfig,
    PDInstrument,
    PDInstrumentConfig,
    SimulationRuntime,
    TaskSpec,
    build_comb_wavelengths,
    simulate_target_resonances,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a bootstrap-guided agent retuning baseline experiment.")
    parser.add_argument("--num-rings", type=int, default=5)
    parser.add_argument("--thermal-tau-ms", type=float, default=8.0)
    parser.add_argument("--settle-ms", type=float, default=40.0)
    parser.add_argument("--controller-rounds", type=int, default=5)
    parser.add_argument("--osa-span-nm", type=float, default=0.4)
    parser.add_argument("--scenario", type=str, default="mild", choices=["mild", "hard"])
    parser.add_argument("--bootstrap-dir", type=str, default="experiments/outputs/baseline_20260311")
    parser.add_argument("--output-dir", type=str, default="experiments/outputs/agent_retuning_baseline")
    return parser.parse_args()


def build_runtime_factory(num_rings: int, thermal_tau_ms: float):
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
                    frame_period_ms=1.0,
                    noise_sigma=0.0,
                ),
                rng=np.random.default_rng(1),
            ),
            osa_instrument=OSAInstrument(
                OSAInstrumentConfig(
                    step_pm=5.0,
                    span_nm=0.8,
                    frame_period_ms=2.0,
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


def build_initial_voltages(target_voltages_v: np.ndarray, scenario: str) -> np.ndarray:
    target_voltages_v = np.asarray(target_voltages_v, dtype=float)
    if scenario == "mild":
        offsets_v = np.linspace(0.4, -0.4, target_voltages_v.shape[0])
    elif scenario == "hard":
        offsets_v = np.array([2.0, 1.2, 0.4, -0.4, -1.2], dtype=float)
        if offsets_v.shape[0] != target_voltages_v.shape[0]:
            offsets_v = np.linspace(2.0, -1.2, target_voltages_v.shape[0])
    else:
        raise ValueError(f"unsupported scenario: {scenario}")
    return np.clip(target_voltages_v + offsets_v, 0.0, 5.0)


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_factory = build_runtime_factory(args.num_rings, args.thermal_tau_ms)
    target_voltages_v = np.linspace(0.8, 2.4, args.num_rings)
    initial_voltages_v = build_initial_voltages(target_voltages_v, args.scenario)
    target_resonances_nm = simulate_target_resonances(
        runtime_factory=runtime_factory,
        target_voltages_v=target_voltages_v,
        settle_ms=args.settle_ms,
    )

    bootstrap_dir = ROOT / args.bootstrap_dir
    calibration_result = None
    calibration_state = None
    if bootstrap_dir.exists():
        calibration_result = CalibrationBootstrap.fit_from_experiment_dir(bootstrap_dir)
        calibration_state = CalibrationState.from_bootstrap_result(
            calibration_result,
            num_rings=args.num_rings,
            confidence=0.9,
        )

    env = AgentEnv(
        runtime_factory=runtime_factory,
        task_spec=TaskSpec(
            target_resonances_nm=target_resonances_nm,
            tolerance_pm=12.0,
            max_episode_time_ms=320.0,
            max_control_steps=80,
            success_hold_steps=1,
            action_budget=100.0,
            observation_budget=120.0,
            allowed_instruments=("OSA", "PD"),
        ),
        budget_config=BudgetConfig(
            voltage_action_cost=1.0,
            wait_cost_per_ms=0.05,
            pd_read_cost=1.0,
            osa_read_cost=4.0,
        ),
    )
    env.reset(initial_voltages_v=initial_voltages_v, settle_ms=args.settle_ms)
    initial_summary = env.episode_summary()

    controller = BootstrapRetuningController(
        calibration_state=calibration_state,
        config=BootstrapRetuningControllerConfig(
            osa_span_nm=args.osa_span_nm,
            settle_ms=args.settle_ms,
            max_rounds=args.controller_rounds,
        ),
    )
    summary = controller.run_episode(env)

    summary_row = {
        "num_rings": args.num_rings,
        "thermal_tau_ms": args.thermal_tau_ms,
        "settle_ms": args.settle_ms,
        "controller_rounds": args.controller_rounds,
        "osa_span_nm": args.osa_span_nm,
        "scenario": args.scenario,
        "bootstrap_dir": str(bootstrap_dir) if calibration_result is not None else "",
        "calibration_confidence": "" if calibration_state is None else calibration_state.confidence,
        "initial_mean_abs_error_pm": initial_summary["final_mean_abs_error_pm"],
        "initial_max_abs_error_pm": initial_summary["final_max_abs_error_pm"],
        "success": summary["success"],
        "done_reason": summary["done_reason"],
        "elapsed_time_ms": summary["elapsed_time_ms"],
        "final_mean_abs_error_pm": summary["final_mean_abs_error_pm"],
        "final_max_abs_error_pm": summary["final_max_abs_error_pm"],
        "mean_abs_error_improvement_pm": (
            initial_summary["final_mean_abs_error_pm"] - summary["final_mean_abs_error_pm"]
        ),
        "max_abs_error_improvement_pm": (
            initial_summary["final_max_abs_error_pm"] - summary["final_max_abs_error_pm"]
        ),
        "total_budget_cost": summary["total_budget_cost"],
        "num_voltage_commands": summary["num_voltage_commands"],
        "num_wait_actions": summary["num_wait_actions"],
        "num_pd_reads": summary["num_pd_reads"],
        "num_osa_reads": summary["num_osa_reads"],
        "num_clamped_actions": summary["num_clamped_actions"],
        "num_saturated_measurements": summary["num_saturated_measurements"],
        "num_safety_warnings": summary["num_safety_warnings"],
    }
    task_rows = []
    for ring_idx, (initial_voltage_v, target_voltage_v, target_resonance_nm) in enumerate(
        zip(initial_voltages_v, target_voltages_v, target_resonances_nm)
    ):
        task_rows.append(
            {
                "ring_idx": ring_idx,
                "initial_voltage_v": float(initial_voltage_v),
                "target_voltage_v": float(target_voltage_v),
                "target_resonance_nm": float(target_resonance_nm),
            }
        )

    write_csv(output_dir / "summary.csv", [summary_row])
    write_csv(output_dir / "task.csv", task_rows)
    write_csv(output_dir / "trajectory.csv", env.episode_log)

    print(f"saved: {output_dir / 'summary.csv'}")
    print(f"saved: {output_dir / 'task.csv'}")
    print(f"saved: {output_dir / 'trajectory.csv'}")
    print(f"success: {summary['success']}")
    print(f"done reason: {summary['done_reason']}")
    print(f"final max abs error [pm]: {summary['final_max_abs_error_pm']:.3f}")
    print(f"total budget cost: {summary['total_budget_cost']:.3f}")


if __name__ == "__main__":
    main()
