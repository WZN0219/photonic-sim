import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photonic_sim import CalibrationBootstrap  # noqa: E402


def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_calibration_bootstrap_builds_summary_from_experiment_dir(tmp_path):
    exp_dir = tmp_path / "exp"

    _write_csv(
        exp_dir / "step_response.csv",
        [
            "time_ms",
            "channel",
            "requested_voltage_v",
            "accepted_voltage_v",
            "actuator_voltage_v",
            "command_power_mw",
            "thermal_power_mw",
            "shift_nm",
            "effective_resonance_nm",
        ],
        [
            {"time_ms": 0, "channel": 1, "requested_voltage_v": 4, "accepted_voltage_v": 4, "actuator_voltage_v": 0, "command_power_mw": 0, "thermal_power_mw": 0, "shift_nm": 0, "effective_resonance_nm": 1550.0},
            {"time_ms": 5, "channel": 1, "requested_voltage_v": 4, "accepted_voltage_v": 4, "actuator_voltage_v": 4, "command_power_mw": 10, "thermal_power_mw": 4, "shift_nm": 0.04, "effective_resonance_nm": 1550.04},
            {"time_ms": 10, "channel": 1, "requested_voltage_v": 4, "accepted_voltage_v": 4, "actuator_voltage_v": 4, "command_power_mw": 10, "thermal_power_mw": 8, "shift_nm": 0.08, "effective_resonance_nm": 1550.08},
            {"time_ms": 15, "channel": 1, "requested_voltage_v": 4, "accepted_voltage_v": 4, "actuator_voltage_v": 4, "command_power_mw": 10, "thermal_power_mw": 10, "shift_nm": 0.1, "effective_resonance_nm": 1550.1},
        ],
    )

    _write_csv(
        exp_dir / "crosstalk_scan.csv",
        ["drive_channel", "drive_voltage_v", "settle_ms", "shift_ring_0_nm", "shift_ring_1_nm", "shift_ring_2_nm"],
        [
            {"drive_channel": 1, "drive_voltage_v": 1.0, "settle_ms": 10.0, "shift_ring_0_nm": 0.002, "shift_ring_1_nm": 0.02, "shift_ring_2_nm": 0.002},
            {"drive_channel": 1, "drive_voltage_v": 2.0, "settle_ms": 10.0, "shift_ring_0_nm": 0.004, "shift_ring_1_nm": 0.04, "shift_ring_2_nm": 0.004},
        ],
    )

    _write_csv(
        exp_dir / "observation_chain" / "pd_sweep.csv",
        ["adc_bits", "full_scale_current_ma", "input_power_mw", "mean_quantized_current_ma", "max_quantized_current_ma", "nonzero_fraction", "saturated", "adc_lsb_ma"],
        [
            {"adc_bits": 4, "full_scale_current_ma": 0.02, "input_power_mw": 1.0, "mean_quantized_current_ma": 0.01, "max_quantized_current_ma": 0.02, "nonzero_fraction": 1.0, "saturated": "True", "adc_lsb_ma": 0.001},
            {"adc_bits": 8, "full_scale_current_ma": 0.2, "input_power_mw": 1.0, "mean_quantized_current_ma": 0.03, "max_quantized_current_ma": 0.04, "nonzero_fraction": 1.0, "saturated": "False", "adc_lsb_ma": 0.0005},
        ],
    )

    _write_csv(
        exp_dir / "observation_chain" / "osa_sweep.csv",
        ["step_pm", "span_nm", "frame_period_ms", "num_samples", "min_spectrum_dbm", "max_spectrum_dbm", "second_frame_quality"],
        [
            {"step_pm": 20.0, "span_nm": 0.8, "frame_period_ms": 5.0, "num_samples": 41, "min_spectrum_dbm": -10, "max_spectrum_dbm": -2, "second_frame_quality": "stale"},
            {"step_pm": 10.0, "span_nm": 0.8, "frame_period_ms": 5.0, "num_samples": 81, "min_spectrum_dbm": -10, "max_spectrum_dbm": -2, "second_frame_quality": "stale"},
        ],
    )

    _write_csv(
        exp_dir / "drift_dataset" / "latent_state.csv",
        ["time_ms", "res_nm_0", "res_nm_1"],
        [
            {"time_ms": 0, "res_nm_0": 1549.90, "res_nm_1": 1550.10},
            {"time_ms": 5, "res_nm_0": 1549.905, "res_nm_1": 1550.11},
            {"time_ms": 10, "res_nm_0": 1549.91, "res_nm_1": 1550.12},
        ],
    )

    _write_csv(
        exp_dir / "drift_dataset" / "pd_frames.csv",
        ["timestamp_ms", "pd_q_ma_0", "pd_q_ma_1", "saturated"],
        [
            {"timestamp_ms": 0, "pd_q_ma_0": 0.01, "pd_q_ma_1": 0.02, "saturated": "False"},
            {"timestamp_ms": 5, "pd_q_ma_0": 0.011, "pd_q_ma_1": 0.021, "saturated": "False"},
            {"timestamp_ms": 10, "pd_q_ma_0": 0.012, "pd_q_ma_1": 0.022, "saturated": "False"},
        ],
    )

    _write_csv(
        exp_dir / "drift_dataset" / "osa_frames.csv",
        ["timestamp_ms", "wavelength_nm", "spectrum_dbm"],
        [
            {"timestamp_ms": 0, "wavelength_nm": 1549.9, "spectrum_dbm": -5.0},
            {"timestamp_ms": 0, "wavelength_nm": 1550.0, "spectrum_dbm": -4.0},
            {"timestamp_ms": 10, "wavelength_nm": 1549.9, "spectrum_dbm": -5.1},
            {"timestamp_ms": 10, "wavelength_nm": 1550.0, "spectrum_dbm": -4.1},
        ],
    )

    result = CalibrationBootstrap.fit_from_experiment_dir(exp_dir)
    assert round(result.step_response.estimated_tuning_efficiency_nm_per_mw, 6) == 0.01
    assert result.crosstalk.drive_channel == 1
    assert round(result.crosstalk.relative_profile_by_offset[0], 6) == 1.0
    assert result.observation.recommended_pd_config["adc_bits"] == 8
    assert result.observation.recommended_osa_config["step_pm"] == 10.0
    assert round(result.drift.pd_frame_period_ms, 6) == 5.0
    assert round(result.drift.osa_frame_period_ms, 6) == 10.0

    output_json = tmp_path / "calibration_bootstrap.json"
    result.save_json(output_json)
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["step_response"]["final_shift_nm"] == 0.1
