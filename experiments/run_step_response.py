import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photonic_sim import ActionExecutorConfig, MRRArrayPlant, MRRPlantConfig, build_comb_wavelengths  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single-channel actuator step response experiment.")
    parser.add_argument("--num-rings", type=int, default=5)
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--target-voltage", type=float, default=4.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--thermal-tau-ms", type=float, default=8.0)
    parser.add_argument("--slew-rate-v-per-ms", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="experiments/outputs/step_response.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=args.num_rings)
    plant = MRRArrayPlant(
        num_rings=args.num_rings,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(
            thermal_tau_ms=args.thermal_tau_ms,
            drift_sigma_nm_per_s=0.0,
        ),
        rng=np.random.default_rng(0),
        action_config=ActionExecutorConfig(slew_rate_v_per_ms=args.slew_rate_v_per_ms),
    )

    ack = plant.issue_command(channel=args.channel, target_voltage_v=args.target_voltage)
    rows = []
    rows.append(
        {
            "time_ms": plant.time_ms,
            "channel": args.channel,
            "requested_voltage_v": ack.requested_voltage_v,
            "accepted_voltage_v": ack.target_voltage_v,
            "actuator_voltage_v": plant.latent_state().actuator_voltages_v[args.channel],
            "command_power_mw": plant.latent_state().command_powers_mw[args.channel],
            "thermal_power_mw": plant.latent_state().thermal_powers_mw[args.channel],
            "shift_nm": plant.latent_state().ideal_shifts_nm[args.channel],
            "effective_resonance_nm": plant.latent_state().effective_resonances_nm[args.channel],
        }
    )

    for _ in range(args.steps):
        state = plant.step(args.dt_ms)
        rows.append(
            {
                "time_ms": state.time_ms,
                "channel": args.channel,
                "requested_voltage_v": ack.requested_voltage_v,
                "accepted_voltage_v": ack.target_voltage_v,
                "actuator_voltage_v": state.actuator_voltages_v[args.channel],
                "command_power_mw": state.command_powers_mw[args.channel],
                "thermal_power_mw": state.thermal_powers_mw[args.channel],
                "shift_nm": state.ideal_shifts_nm[args.channel],
                "effective_resonance_nm": state.effective_resonances_nm[args.channel],
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {output_path}")
    print(f"final shift [nm]: {rows[-1]['shift_nm']:.6f}")
    print(f"final effective resonance [nm]: {rows[-1]['effective_resonance_nm']:.6f}")


if __name__ == "__main__":
    main()
