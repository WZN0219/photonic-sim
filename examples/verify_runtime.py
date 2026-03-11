import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photonic_sim import (  # noqa: E402
    ActionExecutorConfig,
    MRRArrayPlant,
    MRRPlantConfig,
    OSAInstrument,
    OSAInstrumentConfig,
    PDInstrument,
    PDInstrumentConfig,
    SafetyGuardConfig,
    SimulationRuntime,
    build_comb_wavelengths,
)


def title(text: str) -> None:
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def verify_step_and_thermal_dynamics() -> None:
    title("Verification 1: voltage -> power -> thermal state -> shift")

    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=3)
    plant = MRRArrayPlant(
        num_rings=3,
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

    ack = plant.issue_command(channel=1, target_voltage_v=4.0)
    print(
        f"ACK: ch={ack.channel}, requested={ack.requested_voltage_v:.3f}V, "
        f"accepted={ack.target_voltage_v:.3f}V, status={ack.status}, msg={ack.message}"
    )

    for dt_ms in [0.0, 1.0, 2.0, 5.0, 20.0]:
        state = plant.step(dt_ms)
        print(
            f"t={state.time_ms:6.1f} ms | "
            f"V={state.actuator_voltages_v[1]:6.3f} V | "
            f"P_cmd={state.command_powers_mw[1]:8.4f} mW | "
            f"P_th={state.thermal_powers_mw[1]:8.4f} mW | "
            f"shift={state.ideal_shifts_nm[1]:8.5f} nm | "
            f"res={state.effective_resonances_nm[1]:8.5f} nm"
        )


def verify_pd_and_osa_frames() -> None:
    title("Verification 2: timestamped PD / OSA frames")

    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=3)
    plant = MRRArrayPlant(
        num_rings=3,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(
            thermal_tau_ms=6.0,
            drift_sigma_nm_per_s=0.0,
        ),
        rng=np.random.default_rng(1),
        action_config=ActionExecutorConfig(slew_rate_v_per_ms=10.0),
    )

    runtime = SimulationRuntime(
        plant=plant,
        pd_instrument=PDInstrument(
            PDInstrumentConfig(
                adc_bits=8,
                full_scale_current_ma=0.05,
                frame_period_ms=2.0,
                noise_sigma=0.0,
            ),
            rng=np.random.default_rng(2),
        ),
        osa_instrument=OSAInstrument(
            OSAInstrumentConfig(
                step_pm=20.0,
                span_nm=0.8,
                frame_period_ms=5.0,
                amplitude_noise_sigma=0.0,
            ),
            rng=np.random.default_rng(3),
        ),
    )

    runtime.apply_voltage(channel=0, voltage_v=3.5)
    runtime.step(3.0)

    pd_fresh = runtime.read_pd(input_powers_mw=np.full(3, 10.0))
    print(
        f"PD fresh: t={pd_fresh.timestamp_ms:.1f} ms | "
        f"quality={pd_fresh.quality_flag} | calib={pd_fresh.calib_version}"
    )
    print("PD quantized currents [mA]:", np.round(pd_fresh.payload["quantized_currents_ma"], 6))
    print("PD metadata:", pd_fresh.metadata)

    runtime.step(1.0)
    pd_stale = runtime.read_pd(input_powers_mw=np.full(3, 10.0))
    print(
        f"PD second read: t={pd_stale.timestamp_ms:.1f} ms | "
        f"quality={pd_stale.quality_flag} | source_t={pd_stale.metadata.get('source_timestamp_ms')}"
    )

    osa_frame = runtime.read_osa()
    print(
        f"OSA: t={osa_frame.timestamp_ms:.1f} ms | "
        f"quality={osa_frame.quality_flag} | calib={osa_frame.calib_version}"
    )
    print("OSA first 6 wavelengths [nm]:", np.round(osa_frame.payload["wavelengths_nm"][:6], 4))
    print("OSA first 6 spectrum [dBm]:", np.round(osa_frame.payload["spectrum_dbm"][:6], 4))


def verify_executor_and_safety() -> None:
    title("Verification 3: executor clamp and safety warning")

    comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=2)
    plant = MRRArrayPlant(
        num_rings=2,
        comb_wavelengths_nm=comb,
        config=MRRPlantConfig(
            thermal_tau_ms=1.0,
            drift_sigma_nm_per_s=0.0,
        ),
        rng=np.random.default_rng(4),
        action_config=ActionExecutorConfig(
            max_voltage_v=3.0,
            slew_rate_v_per_ms=10.0,
        ),
        safety_config=SafetyGuardConfig(
            max_self_shift_nm=0.02,
            max_total_shift_nm=0.015,
        ),
    )

    ack = plant.issue_command(channel=0, target_voltage_v=6.0)
    print(
        f"ACK: requested={ack.requested_voltage_v:.3f}V, "
        f"accepted={ack.target_voltage_v:.3f}V, status={ack.status}, msg={ack.message}"
    )

    state = plant.step(10.0)
    print("self-shift clamp mask:", state.self_shift_clamped_mask)
    print("total-shift warning mask:", state.total_shift_warning_mask)
    print("ideal shifts [nm]:", np.round(state.ideal_shifts_nm, 6))
    print("effective resonances [nm]:", np.round(state.effective_resonances_nm, 6))


if __name__ == "__main__":
    verify_step_and_thermal_dynamics()
    verify_pd_and_osa_frames()
    verify_executor_and_safety()
