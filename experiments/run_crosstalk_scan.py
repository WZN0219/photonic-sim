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
    parser = argparse.ArgumentParser(description="Scan thermal crosstalk by driving one ring and observing all rings.")
    parser.add_argument("--num-rings", type=int, default=9)
    parser.add_argument("--drive-channel", type=int, default=4)
    parser.add_argument("--max-voltage", type=float, default=4.0)
    parser.add_argument("--num-points", type=int, default=9)
    parser.add_argument("--settle-ms", type=float, default=30.0)
    parser.add_argument("--thermal-tau-ms", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="experiments/outputs/crosstalk_scan.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=args.num_rings)
    voltage_points = np.linspace(0.0, args.max_voltage, args.num_points)
    rows = []

    for drive_voltage in voltage_points:
        plant = MRRArrayPlant(
            num_rings=args.num_rings,
            comb_wavelengths_nm=comb,
            config=MRRPlantConfig(
                thermal_tau_ms=args.thermal_tau_ms,
                drift_sigma_nm_per_s=0.0,
            ),
            rng=np.random.default_rng(0),
            action_config=ActionExecutorConfig(slew_rate_v_per_ms=10.0),
        )
        plant.issue_command(args.drive_channel, float(drive_voltage))
        plant.step(args.settle_ms)
        state = plant.latent_state()
        shifts_nm = state.effective_resonances_nm - plant.base_resonances_nm

        row = {
            "drive_channel": args.drive_channel,
            "drive_voltage_v": float(drive_voltage),
            "settle_ms": args.settle_ms,
        }
        for i in range(args.num_rings):
            row[f"shift_ring_{i}_nm"] = shifts_nm[i]
        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {output_path}")
    max_row = rows[-1]
    print("max-drive shift summary [nm]:")
    for i in range(args.num_rings):
        print(f"  ring {i}: {max_row[f'shift_ring_{i}_nm']:.6f}")


if __name__ == "__main__":
    main()
