import copy
from typing import Optional

import numpy as np

from .config import OSAInstrumentConfig, PDInstrumentConfig
from .physics import adc_quantize_unipolar
from .types import MeasurementFrame


def _clone_frame(frame: Optional[MeasurementFrame]) -> Optional[MeasurementFrame]:
    if frame is None:
        return None
    payload = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else copy.deepcopy(value)
        for key, value in frame.payload.items()
    }
    metadata = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else copy.deepcopy(value)
        for key, value in frame.metadata.items()
    }
    return MeasurementFrame(
        instrument_type=frame.instrument_type,
        timestamp_ms=frame.timestamp_ms,
        calib_version=frame.calib_version,
        quality_flag=frame.quality_flag,
        payload=payload,
        metadata=metadata,
    )


class PDInstrument:
    def __init__(self, config: Optional[PDInstrumentConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self.config = config or PDInstrumentConfig()
        self.rng = rng or np.random.default_rng()
        self._last_frame: Optional[MeasurementFrame] = None

    def snapshot(self) -> dict:
        return {
            "config": self.config,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "last_frame": _clone_frame(self._last_frame),
        }

    def restore(self, snapshot: dict) -> None:
        self.config = snapshot["config"]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        self._last_frame = _clone_frame(snapshot["last_frame"])

    def fork(self) -> "PDInstrument":
        clone = PDInstrument(config=self.config, rng=np.random.default_rng())
        clone.restore(self.snapshot())
        return clone

    def sample(self, plant, input_powers_mw: Optional[np.ndarray] = None,
               wavelengths_nm: Optional[np.ndarray] = None) -> MeasurementFrame:
        now_ms = plant.time_ms
        if self._last_frame is not None and now_ms - self._last_frame.timestamp_ms < self.config.frame_period_ms:
            stale_frame = _clone_frame(self._last_frame)
            stale_metadata = stale_frame.metadata
            stale_metadata.update(
                {
                    "source_timestamp_ms": self._last_frame.timestamp_ms,
                    "frame_period_ms": self.config.frame_period_ms,
                }
            )
            return MeasurementFrame(
                instrument_type="PD",
                timestamp_ms=now_ms,
                calib_version=self.config.calib_version,
                quality_flag="stale",
                payload=stale_frame.payload,
                metadata=stale_metadata,
            )

        wavelengths = plant.comb_wavelengths_nm if wavelengths_nm is None else np.asarray(wavelengths_nm, dtype=float)
        if input_powers_mw is None:
            input_powers_mw = np.ones_like(wavelengths)
        else:
            input_powers_mw = np.asarray(input_powers_mw, dtype=float)

        through_transmission = plant.total_through_transmission(wavelengths)
        optical_powers_mw = input_powers_mw[: wavelengths.shape[0]] * through_transmission
        analog_currents_ma = self.config.responsivity_aw * optical_powers_mw
        analog_currents_ma += self.config.dark_current_na * 1e-6
        analog_currents_ma += self.rng.normal(
            0.0,
            self.config.noise_sigma * np.maximum(analog_currents_ma, 1e-6),
            size=analog_currents_ma.shape,
        )
        analog_currents_ma = np.clip(analog_currents_ma, 0.0, None)

        quantized_currents_ma, lsb_ma = adc_quantize_unipolar(
            analog_currents_ma,
            self.config.adc_bits,
            self.config.full_scale_current_ma,
        )

        frame = MeasurementFrame(
            instrument_type="PD",
            timestamp_ms=now_ms,
            calib_version=self.config.calib_version,
            quality_flag="fresh",
            payload={
                "wavelengths_nm": wavelengths.copy(),
                "input_powers_mw": input_powers_mw[: wavelengths.shape[0]].copy(),
                "optical_powers_mw": optical_powers_mw,
                "analog_currents_ma": analog_currents_ma,
                "quantized_currents_ma": quantized_currents_ma,
            },
            metadata={
                "adc_bits": self.config.adc_bits,
                "adc_full_scale_current_ma": self.config.full_scale_current_ma,
                "adc_lsb_ma": lsb_ma,
                "saturated": bool(np.any(analog_currents_ma > self.config.full_scale_current_ma)),
            },
        )
        self._last_frame = frame
        return frame


class OSAInstrument:
    def __init__(self, config: Optional[OSAInstrumentConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self.config = config or OSAInstrumentConfig()
        self.rng = rng or np.random.default_rng()
        self._last_frame: Optional[MeasurementFrame] = None

    def snapshot(self) -> dict:
        return {
            "config": self.config,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "last_frame": _clone_frame(self._last_frame),
        }

    def restore(self, snapshot: dict) -> None:
        self.config = snapshot["config"]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        self._last_frame = _clone_frame(snapshot["last_frame"])

    def fork(self) -> "OSAInstrument":
        clone = OSAInstrument(config=self.config, rng=np.random.default_rng())
        clone.restore(self.snapshot())
        return clone

    def sample(self, plant, center_nm: Optional[float] = None,
               span_nm: Optional[float] = None) -> MeasurementFrame:
        now_ms = plant.time_ms
        if self._last_frame is not None and now_ms - self._last_frame.timestamp_ms < self.config.frame_period_ms:
            stale_frame = _clone_frame(self._last_frame)
            stale_metadata = stale_frame.metadata
            stale_metadata.update(
                {
                    "source_timestamp_ms": self._last_frame.timestamp_ms,
                    "frame_period_ms": self.config.frame_period_ms,
                }
            )
            return MeasurementFrame(
                instrument_type="OSA",
                timestamp_ms=now_ms,
                calib_version=self.config.calib_version,
                quality_flag="stale",
                payload=stale_frame.payload,
                metadata=stale_metadata,
            )

        center_nm = float(center_nm if center_nm is not None else np.mean(plant.base_resonances_nm))
        span_nm = float(span_nm if span_nm is not None else self.config.span_nm)
        step_nm = self.config.step_pm * 1e-3
        start_nm = center_nm - span_nm / 2.0
        stop_nm = center_nm + span_nm / 2.0 + step_nm / 2.0
        wavelengths_nm = np.arange(start_nm, stop_nm, step_nm)

        through_transmission = plant.total_through_transmission(wavelengths_nm)
        spectrum_mw = np.maximum(through_transmission, self.config.noise_floor_mw)
        spectrum_mw += self.rng.normal(0.0, self.config.amplitude_noise_sigma, size=spectrum_mw.shape)
        spectrum_mw = np.clip(spectrum_mw, self.config.noise_floor_mw, None)
        spectrum_dbm = 10.0 * np.log10(np.maximum(spectrum_mw, 1e-12))

        frame = MeasurementFrame(
            instrument_type="OSA",
            timestamp_ms=now_ms,
            calib_version=self.config.calib_version,
            quality_flag="fresh",
            payload={
                "wavelengths_nm": wavelengths_nm,
                "through_transmission": through_transmission,
                "spectrum_mw": spectrum_mw,
                "spectrum_dbm": spectrum_dbm,
            },
            metadata={
                "center_nm": center_nm,
                "span_nm": span_nm,
                "step_pm": self.config.step_pm,
            },
        )
        self._last_frame = frame
        return frame
