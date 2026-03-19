import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Union


def _read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(value) -> float:
    return float(value)


@dataclass
class StepResponseCalibration:
    final_shift_nm: float
    final_effective_resonance_nm: float
    final_command_power_mw: float
    final_thermal_power_mw: float
    estimated_tuning_efficiency_nm_per_mw: float
    t63_ms: float
    t95_ms: float


@dataclass
class CrosstalkCalibration:
    drive_channel: int
    center_shift_nm: float
    relative_profile_by_offset: Dict[int, float]
    estimated_crosstalk_matrix: List[List[float]]


@dataclass
class ObservationCalibration:
    recommended_pd_config: dict
    recommended_osa_config: dict
    pd_summary: dict
    osa_summary: dict


@dataclass
class DriftCalibration:
    duration_ms: float
    latent_rows: int
    pd_rows: int
    osa_rows: int
    pd_frame_period_ms: float
    osa_frame_period_ms: float
    resonance_span_pm_by_ring: Dict[int, float]


@dataclass
class CalibrationBootstrapResult:
    source_dir: str
    step_response: StepResponseCalibration
    crosstalk: CrosstalkCalibration
    observation: ObservationCalibration
    drift: DriftCalibration

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class CalibrationBootstrap:
    """Build a minimal calibration summary from the baseline experiment CSV outputs."""

    @classmethod
    def fit_from_experiment_dir(cls, experiment_dir: Union[str, Path]) -> CalibrationBootstrapResult:
        experiment_dir = Path(experiment_dir)
        step = cls.fit_step_response(experiment_dir / "step_response.csv")
        crosstalk = cls.fit_crosstalk(experiment_dir / "crosstalk_scan.csv")
        observation = cls.fit_observation_chain(experiment_dir / "observation_chain")
        drift = cls.fit_drift_dataset(experiment_dir / "drift_dataset")
        return CalibrationBootstrapResult(
            source_dir=str(experiment_dir),
            step_response=step,
            crosstalk=crosstalk,
            observation=observation,
            drift=drift,
        )

    @staticmethod
    def fit_step_response(csv_path: Union[str, Path]) -> StepResponseCalibration:
        rows = _read_csv_rows(Path(csv_path))
        if not rows:
            raise ValueError(f"empty step response file: {csv_path}")

        final_row = rows[-1]
        final_shift = _safe_float(final_row["shift_nm"])
        final_thermal_power = _safe_float(final_row["thermal_power_mw"])
        final_command_power = _safe_float(final_row["command_power_mw"])
        t63_threshold = 0.632 * final_shift
        t95_threshold = 0.95 * final_shift
        t63_ms = None
        t95_ms = None

        for row in rows:
            time_ms = _safe_float(row["time_ms"])
            shift_nm = _safe_float(row["shift_nm"])
            if t63_ms is None and shift_nm >= t63_threshold:
                t63_ms = time_ms
            if t95_ms is None and shift_nm >= t95_threshold:
                t95_ms = time_ms
                break

        return StepResponseCalibration(
            final_shift_nm=final_shift,
            final_effective_resonance_nm=_safe_float(final_row["effective_resonance_nm"]),
            final_command_power_mw=final_command_power,
            final_thermal_power_mw=final_thermal_power,
            estimated_tuning_efficiency_nm_per_mw=(
                final_shift / final_thermal_power if final_thermal_power > 0 else 0.0
            ),
            t63_ms=float(t63_ms or 0.0),
            t95_ms=float(t95_ms or 0.0),
        )

    @staticmethod
    def fit_crosstalk(csv_path: Union[str, Path]) -> CrosstalkCalibration:
        rows = _read_csv_rows(Path(csv_path))
        if not rows:
            raise ValueError(f"empty crosstalk file: {csv_path}")

        max_row = max(rows, key=lambda r: _safe_float(r["drive_voltage_v"]))
        drive_channel = int(max_row["drive_channel"])
        shift_items = {
            int(key.replace("shift_ring_", "").replace("_nm", "")): _safe_float(value)
            for key, value in max_row.items()
            if key.startswith("shift_ring_")
        }
        num_rings = len(shift_items)
        center_shift = shift_items[drive_channel]
        if center_shift <= 0:
            raise ValueError("center shift must be positive to estimate crosstalk profile")

        relative_by_offset = {}
        for ring_idx, shift_nm in shift_items.items():
            relative_by_offset[ring_idx - drive_channel] = shift_nm / center_shift

        profile_by_abs_offset = {}
        for offset, ratio in relative_by_offset.items():
            profile_by_abs_offset[abs(offset)] = ratio

        estimated_matrix = []
        for i in range(num_rings):
            row = []
            for j in range(num_rings):
                row.append(float(profile_by_abs_offset.get(abs(i - j), 0.0)))
            estimated_matrix.append(row)

        return CrosstalkCalibration(
            drive_channel=drive_channel,
            center_shift_nm=center_shift,
            relative_profile_by_offset={int(k): float(v) for k, v in sorted(relative_by_offset.items())},
            estimated_crosstalk_matrix=estimated_matrix,
        )

    @staticmethod
    def fit_observation_chain(observation_dir: Union[str, Path]) -> ObservationCalibration:
        observation_dir = Path(observation_dir)
        pd_rows = _read_csv_rows(observation_dir / "pd_sweep.csv")
        osa_rows = _read_csv_rows(observation_dir / "osa_sweep.csv")

        def pd_score(row: dict):
            saturated = row["saturated"] == "True"
            return (
                int(not saturated),
                -abs(_safe_float(row["input_power_mw"]) - 1.0),
                _safe_float(row["nonzero_fraction"]),
                _safe_float(row["mean_quantized_current_ma"]),
                _safe_float(row["adc_bits"]),
            )

        recommended_pd = max(pd_rows, key=pd_score)

        def osa_score(row: dict):
            if "fresh_post_step_mean_peak_error_pm" in row:
                fresh_quality = row.get("fresh_post_step_quality", "")
                early_quality = row.get("early_post_step_quality", "")
                return (
                    int(fresh_quality == "fresh"),
                    int(early_quality == "stale"),
                    -_safe_float(row["fresh_post_step_mean_peak_error_pm"]),
                    -_safe_float(row["fresh_post_step_max_peak_error_pm"]),
                    -_safe_float(row["frame_period_ms"]),
                    -_safe_float(row["num_samples"]),
                )
            num_samples = _safe_float(row["num_samples"])
            frame_period = _safe_float(row["frame_period_ms"])
            info_rate = num_samples / frame_period if frame_period > 0 else 0.0
            return (
                info_rate,
                -_safe_float(row["step_pm"]),
                num_samples,
            )

        recommended_osa = max(osa_rows, key=osa_score)

        pd_summary = {
            "total_rows": len(pd_rows),
            "num_saturated_rows": sum(1 for row in pd_rows if row["saturated"] == "True"),
            "num_full_nonzero_rows": sum(1 for row in pd_rows if _safe_float(row["nonzero_fraction"]) >= 1.0),
        }
        osa_summary = {
            "total_rows": len(osa_rows),
            "num_stale_second_frame_rows": sum(
                1
                for row in osa_rows
                if row.get("second_frame_quality") == "stale"
                or row.get("early_post_step_quality") == "stale"
            ),
            "num_fresh_post_step_rows": sum(
                1 for row in osa_rows if row.get("fresh_post_step_quality") == "fresh"
            ),
            "unique_num_samples": sorted(
                {int(_safe_float(row["num_samples"])) for row in osa_rows}
            ),
        }

        recommended_osa_config = {
            "step_pm": _safe_float(recommended_osa["step_pm"]),
            "span_nm": _safe_float(recommended_osa["span_nm"]),
            "frame_period_ms": _safe_float(recommended_osa["frame_period_ms"]),
            "num_samples": int(_safe_float(recommended_osa["num_samples"])),
        }
        if "fresh_post_step_quality" in recommended_osa:
            recommended_osa_config.update(
                {
                    "fresh_post_step_quality": recommended_osa["fresh_post_step_quality"],
                    "early_post_step_quality": recommended_osa["early_post_step_quality"],
                    "fresh_post_step_mean_peak_error_pm": _safe_float(
                        recommended_osa["fresh_post_step_mean_peak_error_pm"]
                    ),
                    "fresh_post_step_max_peak_error_pm": _safe_float(
                        recommended_osa["fresh_post_step_max_peak_error_pm"]
                    ),
                }
            )

        return ObservationCalibration(
            recommended_pd_config={
                "adc_bits": int(_safe_float(recommended_pd["adc_bits"])),
                "full_scale_current_ma": _safe_float(recommended_pd["full_scale_current_ma"]),
                "input_power_mw": _safe_float(recommended_pd["input_power_mw"]),
                "nonzero_fraction": _safe_float(recommended_pd["nonzero_fraction"]),
                "saturated": recommended_pd["saturated"] == "True",
                "adc_lsb_ma": _safe_float(recommended_pd["adc_lsb_ma"]),
            },
            recommended_osa_config=recommended_osa_config,
            pd_summary=pd_summary,
            osa_summary=osa_summary,
        )

    @staticmethod
    def fit_drift_dataset(drift_dir: Union[str, Path]) -> DriftCalibration:
        drift_dir = Path(drift_dir)
        latent_rows = _read_csv_rows(drift_dir / "latent_state.csv")
        pd_rows = _read_csv_rows(drift_dir / "pd_frames.csv")
        osa_rows = _read_csv_rows(drift_dir / "osa_frames.csv")

        if not latent_rows:
            raise ValueError(f"empty latent_state.csv in {drift_dir}")

        resonance_cols = [col for col in latent_rows[0].keys() if col.startswith("res_nm_")]
        resonance_span_pm_by_ring = {}
        for col in resonance_cols:
            ring_idx = int(col.replace("res_nm_", ""))
            values = [_safe_float(row[col]) for row in latent_rows]
            resonance_span_pm_by_ring[ring_idx] = (max(values) - min(values)) * 1000.0

        pd_frame_period_ms = 0.0
        if len(pd_rows) > 1:
            pd_frame_period_ms = (
                _safe_float(pd_rows[-1]["timestamp_ms"]) - _safe_float(pd_rows[0]["timestamp_ms"])
            ) / (len(pd_rows) - 1)

        osa_timestamps = sorted({_safe_float(row["timestamp_ms"]) for row in osa_rows})
        osa_frame_period_ms = 0.0
        if len(osa_timestamps) > 1:
            osa_frame_period_ms = (osa_timestamps[-1] - osa_timestamps[0]) / (len(osa_timestamps) - 1)

        duration_ms = _safe_float(latent_rows[-1]["time_ms"]) - _safe_float(latent_rows[0]["time_ms"])

        return DriftCalibration(
            duration_ms=duration_ms,
            latent_rows=len(latent_rows),
            pd_rows=len(pd_rows),
            osa_rows=len(osa_rows),
            pd_frame_period_ms=pd_frame_period_ms,
            osa_frame_period_ms=osa_frame_period_ms,
            resonance_span_pm_by_ring=resonance_span_pm_by_ring,
        )
