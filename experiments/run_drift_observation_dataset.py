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
    SimulationRuntime,
    build_comb_wavelengths,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate long-horizon drift observation data.")
    parser.add_argument("--num-rings", type=int, default=5)
    parser.add_argument("--duration-ms", type=float, default=200.0)
    parser.add_argument("--step-ms", type=float, default=1.0)
    parser.add_argument("--drift-sigma-nm-per-s", type=float, default=0.002)
    parser.add_argument("--output-dir", type=str, default="experiments/outputs/drift_dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=args.num_rings)
    plant = MRRArrayPlant(
        num_rings=args.num_rings,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(
            thermal_tau_ms=6.0,
            drift_sigma_nm_per_s=args.drift_sigma_nm_per_s,
        ),
        rng=np.random.default_rng(0),
        action_config=ActionExecutorConfig(slew_rate_v_per_ms=5.0),
    )
    runtime = SimulationRuntime(
        plant=plant,
        pd_instrument=PDInstrument(
            PDInstrumentConfig(
                adc_bits=10,
                full_scale_current_ma=0.2,
                frame_period_ms=5.0,
                noise_sigma=0.001,
            ),
            rng=np.random.default_rng(1),
        ),
        osa_instrument=OSAInstrument(
            OSAInstrumentConfig(
                step_pm=10.0,
                span_nm=0.8,
                frame_period_ms=20.0,
                amplitude_noise_sigma=0.0005,
            ),
            rng=np.random.default_rng(2),
        ),
    )

    for channel, voltage in enumerate(np.linspace(1.0, 3.0, args.num_rings)):
        runtime.apply_voltage(channel, float(voltage))

    latent_rows = []
    pd_rows = []
    osa_rows = []
    last_pd_fresh = None
    last_osa_fresh = None

    num_steps = int(args.duration_ms / args.step_ms)
    for _ in range(num_steps):
        state = runtime.step(args.step_ms)
        latent_row = {"time_ms": state.time_ms}
        for i in range(args.num_rings):
            latent_row[f"actuator_v_{i}"] = state.actuator_voltages_v[i]
            latent_row[f"thermal_power_mw_{i}"] = state.thermal_powers_mw[i]
            latent_row[f"shift_nm_{i}"] = state.ideal_shifts_nm[i]
            latent_row[f"res_nm_{i}"] = state.effective_resonances_nm[i]
        latent_rows.append(latent_row)

        pd_frame = runtime.read_pd(input_powers_mw=np.ones(args.num_rings))
        if pd_frame.quality_flag == "fresh" and pd_frame.timestamp_ms != last_pd_fresh:
            row = {"timestamp_ms": pd_frame.timestamp_ms}
            for i, value in enumerate(pd_frame.payload["quantized_currents_ma"]):
                row[f"pd_q_ma_{i}"] = value
            row["saturated"] = pd_frame.metadata["saturated"]
            pd_rows.append(row)
            last_pd_fresh = pd_frame.timestamp_ms

        osa_frame = runtime.read_osa()
        if osa_frame.quality_flag == "fresh" and osa_frame.timestamp_ms != last_osa_fresh:
            for wavelength_nm, spectrum_dbm in zip(
                osa_frame.payload["wavelengths_nm"],
                osa_frame.payload["spectrum_dbm"],
            ):
                osa_rows.append(
                    {
                        "timestamp_ms": osa_frame.timestamp_ms,
                        "wavelength_nm": wavelength_nm,
                        "spectrum_dbm": spectrum_dbm,
                    }
                )
            last_osa_fresh = osa_frame.timestamp_ms

    latent_path = output_dir / "latent_state.csv"
    with latent_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(latent_rows[0].keys()))
        writer.writeheader()
        writer.writerows(latent_rows)

    pd_path = output_dir / "pd_frames.csv"
    with pd_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pd_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pd_rows)

    osa_path = output_dir / "osa_frames.csv"
    with osa_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(osa_rows[0].keys()))
        writer.writeheader()
        writer.writerows(osa_rows)

    print(f"saved: {latent_path}")
    print(f"saved: {pd_path}")
    print(f"saved: {osa_path}")


if __name__ == "__main__":
    main()
