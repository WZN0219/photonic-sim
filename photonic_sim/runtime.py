import copy
from typing import Optional

from .instruments import OSAInstrument, PDInstrument
from .types import RuntimeSnapshot


class SimulationRuntime:
    """Thin orchestration layer over plant + instruments."""

    def __init__(self, plant, pd_instrument: Optional[PDInstrument] = None,
                 osa_instrument: Optional[OSAInstrument] = None):
        self.plant = plant
        self.pd = pd_instrument
        self.osa = osa_instrument
        self.action_log: list[dict] = []
        self.measurement_log: list[dict] = []

    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            plant_snapshot=self.plant.snapshot(),
            pd_snapshot=None if self.pd is None else self.pd.snapshot(),
            osa_snapshot=None if self.osa is None else self.osa.snapshot(),
            action_log=copy.deepcopy(self.action_log),
            measurement_log=copy.deepcopy(self.measurement_log),
        )

    def restore(self, snapshot: RuntimeSnapshot) -> None:
        self.plant.restore(snapshot.plant_snapshot)
        if snapshot.pd_snapshot is None:
            self.pd = None
        else:
            if self.pd is None:
                self.pd = PDInstrument()
            self.pd.restore(snapshot.pd_snapshot)
        if snapshot.osa_snapshot is None:
            self.osa = None
        else:
            if self.osa is None:
                self.osa = OSAInstrument()
            self.osa.restore(snapshot.osa_snapshot)
        self.action_log = copy.deepcopy(snapshot.action_log)
        self.measurement_log = copy.deepcopy(snapshot.measurement_log)

    def fork(self) -> "SimulationRuntime":
        clone = SimulationRuntime(
            plant=self.plant.fork(),
            pd_instrument=None if self.pd is None else self.pd.fork(),
            osa_instrument=None if self.osa is None else self.osa.fork(),
        )
        clone.action_log = copy.deepcopy(self.action_log)
        clone.measurement_log = copy.deepcopy(self.measurement_log)
        return clone

    def apply_voltage(self, channel: int, voltage_v: float):
        ack = self.plant.issue_command(channel, voltage_v)
        self.action_log.append(
            {
                "timestamp_ms": ack.issued_at_ms,
                "channel": ack.channel,
                "requested_voltage_v": ack.requested_voltage_v,
                "target_voltage_v": ack.target_voltage_v,
                "status": ack.status,
                "message": ack.message,
            }
        )
        return ack

    def step(self, dt_ms: float):
        return self.plant.step(dt_ms)

    def read_pd(self, input_powers_mw=None, wavelengths_nm=None):
        if self.pd is None:
            raise RuntimeError("PD instrument is not configured")
        frame = self.pd.sample(self.plant, input_powers_mw=input_powers_mw, wavelengths_nm=wavelengths_nm)
        self.measurement_log.append(
            {
                "instrument_type": frame.instrument_type,
                "timestamp_ms": frame.timestamp_ms,
                "quality_flag": frame.quality_flag,
                "calib_version": frame.calib_version,
            }
        )
        return frame

    def read_osa(self, center_nm=None, span_nm=None):
        if self.osa is None:
            raise RuntimeError("OSA instrument is not configured")
        frame = self.osa.sample(self.plant, center_nm=center_nm, span_nm=span_nm)
        self.measurement_log.append(
            {
                "instrument_type": frame.instrument_type,
                "timestamp_ms": frame.timestamp_ms,
                "quality_flag": frame.quality_flag,
                "calib_version": frame.calib_version,
            }
        )
        return frame
