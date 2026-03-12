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
    MRRArrayPlant,
    MRRPlantConfig,
    OSAInstrument,
    OSAInstrumentConfig,
    PDInstrument,
    PDInstrumentConfig,
    build_comb_wavelengths,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep PD ADC settings and OSA sampling settings.")
    parser.add_argument("--num-rings", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="experiments/outputs/observation_chain")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=args.num_rings)
    plant = MRRArrayPlant(
        num_rings=args.num_rings,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(thermal_tau_ms=6.0, drift_sigma_nm_per_s=0.0),
        rng=np.random.default_rng(0),
        action_config=ActionExecutorConfig(slew_rate_v_per_ms=10.0),
    )
    plant.issue_command(0, 3.5)
    plant.step(15.0)
    plant.set_global_temp_shift_nm(plant.config.fsr_nm / 2.0)

    pd_rows = []
    adc_bits_options = [4, 6, 8, 10, 12]
    full_scale_options = [0.02, 0.05, 0.2]
    input_power_options = [0.1, 1.0, 10.0]

    for adc_bits in adc_bits_options:
        for full_scale in full_scale_options:
            for input_power in input_power_options:
                pd = PDInstrument(
                    PDInstrumentConfig(
                        adc_bits=adc_bits,
                        full_scale_current_ma=full_scale,
                        frame_period_ms=0.0,
                        noise_sigma=0.0,
                    ),
                    rng=np.random.default_rng(1),
                )
                frame = pd.sample(plant, input_powers_mw=np.full(args.num_rings, input_power))
                quantized = frame.payload["quantized_currents_ma"]
                pd_rows.append(
                    {
                        "adc_bits": adc_bits,
                        "full_scale_current_ma": full_scale,
                        "input_power_mw": input_power,
                        "mean_quantized_current_ma": float(np.mean(quantized)),
                        "max_quantized_current_ma": float(np.max(quantized)),
                        "nonzero_fraction": float(np.mean(quantized > 0)),
                        "saturated": frame.metadata["saturated"],
                        "adc_lsb_ma": frame.metadata["adc_lsb_ma"],
                    }
                )

    pd_path = output_dir / "pd_sweep.csv"
    with pd_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pd_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pd_rows)

    osa_rows = []
    for step_pm in [5.0, 10.0, 20.0]:
        for span_nm in [0.4, 0.8, 1.2]:
            for frame_period_ms in [2.0, 5.0, 10.0]:
                osa = OSAInstrument(
                    OSAInstrumentConfig(
                        step_pm=step_pm,
                        span_nm=span_nm,
                        frame_period_ms=frame_period_ms,
                        amplitude_noise_sigma=0.0,
                    ),
                    rng=np.random.default_rng(2),
                )
                frame_1 = osa.sample(plant)
                plant.step(1.0)
                frame_2 = osa.sample(plant)
                osa_rows.append(
                    {
                        "step_pm": step_pm,
                        "span_nm": span_nm,
                        "frame_period_ms": frame_period_ms,
                        "num_samples": len(frame_1.payload["wavelengths_nm"]),
                        "min_spectrum_dbm": float(np.min(frame_1.payload["spectrum_dbm"])),
                        "max_spectrum_dbm": float(np.max(frame_1.payload["spectrum_dbm"])),
                        "second_frame_quality": frame_2.quality_flag,
                    }
                )

    osa_path = output_dir / "osa_sweep.csv"
    with osa_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(osa_rows[0].keys()))
        writer.writeheader()
        writer.writerows(osa_rows)

    print(f"saved: {pd_path}")
    print(f"saved: {osa_path}")


if __name__ == "__main__":
    main()
