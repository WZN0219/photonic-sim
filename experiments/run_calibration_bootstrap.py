import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photonic_sim import CalibrationBootstrap  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build a minimal calibration bootstrap summary from experiment CSV outputs.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/outputs/baseline_20260311",
        help="Experiment output directory containing step_response.csv, crosstalk_scan.csv, observation_chain/, drift_dataset/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/outputs/baseline_20260311/calibration_bootstrap.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = ROOT / args.input_dir
    output_path = ROOT / args.output

    result = CalibrationBootstrap.fit_from_experiment_dir(input_dir)
    result.save_json(output_path)

    print(f"saved: {output_path}")
    print(f"estimated thermal tau t63 [ms]: {result.step_response.t63_ms:.3f}")
    print(f"estimated tuning efficiency [nm/mW]: {result.step_response.estimated_tuning_efficiency_nm_per_mw:.6f}")
    print(f"recommended PD config: {result.observation.recommended_pd_config}")
    print(f"recommended OSA config: {result.observation.recommended_osa_config}")


if __name__ == "__main__":
    main()
