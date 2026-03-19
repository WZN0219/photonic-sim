import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photonic_sim import (  # noqa: E402
    ActionExecutorConfig,
    MRRArrayPlant,
    MRRPlantConfig,
    build_comb_wavelengths,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a visual overview of an MRR array: resonances, spectrum, and crosstalk."
    )
    parser.add_argument("--num-rings", type=int, default=8)
    parser.add_argument("--center-nm", type=float, default=1550.0)
    parser.add_argument("--fsr-nm", type=float, default=0.73)
    parser.add_argument("--thermal-tau-ms", type=float, default=8.0)
    parser.add_argument("--crosstalk-alpha", type=float, default=0.08)
    parser.add_argument("--crosstalk-decay-length", type=float, default=2.0)
    parser.add_argument("--settle-ms", type=float, default=30.0)
    parser.add_argument("--output-dir", type=str, default="examples/outputs/mrr_visual_overview")
    return parser.parse_args()


def build_demo_voltages(num_rings: int) -> np.ndarray:
    base = np.linspace(0.8, 2.6, num_rings)
    ripple = 0.35 * np.sin(np.linspace(0.0, 2.5 * np.pi, num_rings))
    return np.clip(base + ripple, 0.0, 5.0)


def save_resonance_plot(path: Path, plant: MRRArrayPlant) -> None:
    state = plant.latent_state()
    ring_idx = np.arange(plant.num_rings)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ring_idx, plant.base_resonances_nm, "o--", label="Base resonance")
    ax.plot(ring_idx, state.effective_resonances_nm, "o-", label="Effective resonance")
    ax.set_title("MRR Resonance Positions")
    ax.set_xlabel("Ring index")
    ax.set_ylabel("Wavelength (nm)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_spectrum_plot(path: Path, plant: MRRArrayPlant) -> None:
    margin_nm = max(0.3, 0.15 * plant.num_rings)
    wavelengths_nm = np.linspace(
        float(np.min(plant.base_resonances_nm) - margin_nm),
        float(np.max(plant.base_resonances_nm) + margin_nm),
        2400,
    )
    transmission = plant.total_through_transmission(wavelengths_nm)
    spectrum_db = 10.0 * np.log10(np.maximum(transmission, 1e-12))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(wavelengths_nm, spectrum_db, color="#0f4c81", lw=1.5)
    ax.set_title("Through Spectrum")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission (dB)")
    ax.grid(True, alpha=0.25)

    for value in plant.base_resonances_nm:
        ax.axvline(float(value), color="#999999", lw=0.8, alpha=0.3)

    state = plant.latent_state()
    for value in state.effective_resonances_nm:
        ax.axvline(float(value), color="#c0392b", lw=0.9, alpha=0.45)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_crosstalk_heatmap(path: Path, plant: MRRArrayPlant) -> None:
    matrix = plant.crosstalk_matrix
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title("Crosstalk Matrix")
    ax.set_xlabel("Source ring")
    ax.set_ylabel("Affected ring")
    ax.set_xticks(range(plant.num_rings))
    ax.set_yticks(range(plant.num_rings))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Coupling weight")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_crosstalk_profile(path: Path, plant: MRRArrayPlant) -> None:
    center = plant.num_rings // 2
    offsets = np.arange(plant.num_rings) - center
    profile = plant.crosstalk_matrix[center]

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.bar(offsets, profile, color="#1f77b4")
    ax.set_title(f"Crosstalk Profile From Ring {center}")
    ax.set_xlabel("Ring offset")
    ax.set_ylabel("Coupling weight")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    comb = build_comb_wavelengths(
        center_nm=args.center_nm,
        fsr_nm=args.fsr_nm,
        num_lines=args.num_rings,
    )
    plant = MRRArrayPlant(
        num_rings=args.num_rings,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(
            thermal_tau_ms=args.thermal_tau_ms,
            drift_sigma_nm_per_s=0.0,
            crosstalk_alpha=args.crosstalk_alpha,
            crosstalk_decay_length=args.crosstalk_decay_length,
        ),
        rng=np.random.default_rng(0),
        action_config=ActionExecutorConfig(
            max_voltage_v=5.0,
            slew_rate_v_per_ms=10.0,
        ),
    )

    demo_voltages = build_demo_voltages(args.num_rings)
    for channel, voltage_v in enumerate(demo_voltages):
        plant.issue_command(channel=channel, target_voltage_v=float(voltage_v))
    plant.step(float(args.settle_ms))

    save_resonance_plot(output_dir / "resonance_positions.png", plant)
    save_spectrum_plot(output_dir / "through_spectrum.png", plant)
    save_crosstalk_heatmap(output_dir / "crosstalk_matrix.png", plant)
    save_crosstalk_profile(output_dir / "crosstalk_profile.png", plant)

    print(f"saved: {output_dir / 'resonance_positions.png'}")
    print(f"saved: {output_dir / 'through_spectrum.png'}")
    print(f"saved: {output_dir / 'crosstalk_matrix.png'}")
    print(f"saved: {output_dir / 'crosstalk_profile.png'}")


if __name__ == "__main__":
    main()
