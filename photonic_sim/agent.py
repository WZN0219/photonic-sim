import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .runtime import SimulationRuntime
from .types import ActionAck, AgentEnvSnapshot, BudgetAccountantSnapshot, MeasurementFrame


@dataclass(frozen=True)
class TaskSpec:
    target_resonances_nm: np.ndarray
    tolerance_pm: float = 10.0
    max_episode_time_ms: float = 300.0
    max_control_steps: int = 80
    success_hold_steps: int = 2
    action_budget: float = 80.0
    observation_budget: float = 120.0
    allowed_instruments: tuple[str, ...] = ("PD", "OSA")

    def __post_init__(self) -> None:
        target = np.asarray(self.target_resonances_nm, dtype=float)
        if target.ndim != 1 or target.size == 0:
            raise ValueError("target_resonances_nm must be a non-empty 1D array")
        if self.tolerance_pm <= 0:
            raise ValueError("tolerance_pm must be positive")
        if self.max_episode_time_ms <= 0:
            raise ValueError("max_episode_time_ms must be positive")
        if self.max_control_steps <= 0:
            raise ValueError("max_control_steps must be positive")
        if self.success_hold_steps <= 0:
            raise ValueError("success_hold_steps must be positive")
        object.__setattr__(self, "target_resonances_nm", target.copy())
        object.__setattr__(self, "allowed_instruments", tuple(self.allowed_instruments))


@dataclass(frozen=True)
class BudgetConfig:
    voltage_action_cost: float = 1.0
    wait_cost_per_ms: float = 0.05
    pd_read_cost: float = 1.0
    osa_read_cost: float = 4.0
    clamp_penalty_cost: float = 0.25
    saturation_penalty_cost: float = 0.5
    safety_warning_cost: float = 0.5


@dataclass(frozen=True)
class BudgetSnapshot:
    action_budget_used: float
    observation_budget_used: float
    action_budget_remaining: float
    observation_budget_remaining: float
    total_cost: float
    num_voltage_commands: int
    num_wait_actions: int
    num_pd_reads: int
    num_osa_reads: int
    num_clamped_actions: int
    num_saturated_measurements: int
    num_safety_warnings: int
    num_safety_warning_active_steps: int


class BudgetAccountant:
    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()
        self.reset()

    def reset(self, initial_warning_mask: Optional[np.ndarray] = None) -> None:
        self.action_budget_used = 0.0
        self.observation_budget_used = 0.0
        self.num_voltage_commands = 0
        self.num_wait_actions = 0
        self.num_pd_reads = 0
        self.num_osa_reads = 0
        self.num_clamped_actions = 0
        self.num_saturated_measurements = 0
        self.num_safety_warnings = 0
        self.num_safety_warning_active_steps = 0
        if initial_warning_mask is None:
            self._previous_warning_mask = None
        else:
            self._previous_warning_mask = np.asarray(initial_warning_mask, dtype=bool).copy()

    def charge_voltage(self, ack: ActionAck) -> tuple[float, float]:
        self.action_budget_used += self.config.voltage_action_cost
        self.num_voltage_commands += 1
        penalty = 0.0
        if ack.status == "clamped":
            self.num_clamped_actions += 1
            penalty += self.config.clamp_penalty_cost
        return self.config.voltage_action_cost, penalty

    def charge_wait(self, dt_ms: float) -> tuple[float, float]:
        cost = self.config.wait_cost_per_ms * float(dt_ms)
        self.action_budget_used += cost
        self.num_wait_actions += 1
        return cost, 0.0

    def charge_pd(self, frame: MeasurementFrame) -> tuple[float, float]:
        self.observation_budget_used += self.config.pd_read_cost
        self.num_pd_reads += 1
        penalty = 0.0
        if bool(frame.metadata.get("saturated", False)):
            self.num_saturated_measurements += 1
            penalty += self.config.saturation_penalty_cost
        return self.config.pd_read_cost, penalty

    def charge_osa(self, _frame: MeasurementFrame) -> tuple[float, float]:
        self.observation_budget_used += self.config.osa_read_cost
        self.num_osa_reads += 1
        return self.config.osa_read_cost, 0.0

    def charge_safety(self, warning_mask: np.ndarray) -> float:
        warning_mask = np.asarray(warning_mask, dtype=bool)
        if np.any(warning_mask):
            self.num_safety_warning_active_steps += 1
        if self._previous_warning_mask is None:
            new_warning_mask = warning_mask
        else:
            new_warning_mask = warning_mask & ~self._previous_warning_mask
        num_new_warnings = int(np.count_nonzero(new_warning_mask))
        self.num_safety_warnings += num_new_warnings
        self._previous_warning_mask = warning_mask.copy()
        return num_new_warnings * self.config.safety_warning_cost

    def snapshot(self, task_spec: TaskSpec) -> BudgetSnapshot:
        return BudgetSnapshot(
            action_budget_used=float(self.action_budget_used),
            observation_budget_used=float(self.observation_budget_used),
            action_budget_remaining=float(max(task_spec.action_budget - self.action_budget_used, 0.0)),
            observation_budget_remaining=float(
                max(task_spec.observation_budget - self.observation_budget_used, 0.0)
            ),
            total_cost=float(self.action_budget_used + self.observation_budget_used),
            num_voltage_commands=self.num_voltage_commands,
            num_wait_actions=self.num_wait_actions,
            num_pd_reads=self.num_pd_reads,
            num_osa_reads=self.num_osa_reads,
            num_clamped_actions=self.num_clamped_actions,
            num_saturated_measurements=self.num_saturated_measurements,
            num_safety_warnings=self.num_safety_warnings,
            num_safety_warning_active_steps=self.num_safety_warning_active_steps,
        )

    def exceeded_reason(self, task_spec: TaskSpec) -> Optional[str]:
        if self.action_budget_used > task_spec.action_budget:
            return "action_budget_exceeded"
        if self.observation_budget_used > task_spec.observation_budget:
            return "observation_budget_exceeded"
        return None

    def snapshot_state(self) -> BudgetAccountantSnapshot:
        previous_warning_mask = None if self._previous_warning_mask is None else self._previous_warning_mask.copy()
        return BudgetAccountantSnapshot(
            action_budget_used=float(self.action_budget_used),
            observation_budget_used=float(self.observation_budget_used),
            num_voltage_commands=self.num_voltage_commands,
            num_wait_actions=self.num_wait_actions,
            num_pd_reads=self.num_pd_reads,
            num_osa_reads=self.num_osa_reads,
            num_clamped_actions=self.num_clamped_actions,
            num_saturated_measurements=self.num_saturated_measurements,
            num_safety_warnings=self.num_safety_warnings,
            num_safety_warning_active_steps=self.num_safety_warning_active_steps,
            previous_warning_mask=previous_warning_mask,
        )

    def restore_state(self, snapshot: BudgetAccountantSnapshot) -> None:
        self.action_budget_used = float(snapshot.action_budget_used)
        self.observation_budget_used = float(snapshot.observation_budget_used)
        self.num_voltage_commands = int(snapshot.num_voltage_commands)
        self.num_wait_actions = int(snapshot.num_wait_actions)
        self.num_pd_reads = int(snapshot.num_pd_reads)
        self.num_osa_reads = int(snapshot.num_osa_reads)
        self.num_clamped_actions = int(snapshot.num_clamped_actions)
        self.num_saturated_measurements = int(snapshot.num_saturated_measurements)
        self.num_safety_warnings = int(snapshot.num_safety_warnings)
        self.num_safety_warning_active_steps = int(snapshot.num_safety_warning_active_steps)
        self._previous_warning_mask = None if snapshot.previous_warning_mask is None else snapshot.previous_warning_mask.copy()


@dataclass
class AgentObservation:
    time_ms: float
    task_spec: TaskSpec
    commanded_voltages_v: np.ndarray
    budget_state: BudgetSnapshot
    last_action: Optional[dict[str, Any]]
    last_ack: Optional[ActionAck]
    latest_pd_frame: Optional[MeasurementFrame]
    latest_osa_frame: Optional[MeasurementFrame]
    history_summary: dict[str, Any]


@dataclass
class AgentStepResult:
    observation: AgentObservation
    reward: float
    done: bool
    info: dict[str, Any]


class AgentEnv:
    """Episode wrapper that exposes action, observation, and budget semantics."""

    def __init__(
        self,
        runtime_factory: Callable[[], SimulationRuntime],
        task_spec: TaskSpec,
        budget_config: Optional[BudgetConfig] = None,
    ):
        self.runtime_factory = runtime_factory
        self.task_spec = task_spec
        self.accountant = BudgetAccountant(budget_config)
        self.runtime: Optional[SimulationRuntime] = None
        self.episode_log: list[dict[str, Any]] = []
        self._episode_start_ms = 0.0
        self._step_index = 0
        self._success_streak = 0
        self._previous_mean_abs_error_pm = 0.0
        self._done = False
        self._done_reason = "not_started"
        self._last_action: Optional[dict[str, Any]] = None
        self._last_ack: Optional[ActionAck] = None
        self._latest_pd_frame: Optional[MeasurementFrame] = None
        self._latest_osa_frame: Optional[MeasurementFrame] = None

    def reset(
        self,
        initial_voltages_v: Optional[np.ndarray] = None,
        settle_ms: float = 0.0,
        global_temp_shift_nm: float = 0.0,
        per_ring_static_shift_nm: Optional[np.ndarray] = None,
    ) -> AgentObservation:
        self.runtime = self.runtime_factory()
        if self.task_spec.target_resonances_nm.shape[0] != self.runtime.plant.num_rings:
            raise ValueError("task_spec target_resonances_nm must match plant num_rings")

        if global_temp_shift_nm != 0.0:
            self.runtime.plant.set_global_temp_shift_nm(global_temp_shift_nm)

        if initial_voltages_v is not None:
            initial_voltages_v = np.asarray(initial_voltages_v, dtype=float)
            if initial_voltages_v.shape != (self.runtime.plant.num_rings,):
                raise ValueError("initial_voltages_v must match plant num_rings")
            for channel, voltage_v in enumerate(initial_voltages_v):
                self.runtime.plant.issue_command(channel, float(voltage_v))

        if settle_ms > 0.0:
            self.runtime.step(float(settle_ms))

        if per_ring_static_shift_nm is not None:
            self.runtime.plant.set_per_ring_static_shift_nm(np.asarray(per_ring_static_shift_nm, dtype=float))

        initial_warning_mask = self.runtime.plant.latent_state().total_shift_warning_mask
        self.accountant.reset(initial_warning_mask=initial_warning_mask)
        self.episode_log = []
        self._episode_start_ms = self.runtime.plant.time_ms
        self._step_index = 0
        self._success_streak = 0
        self._done = False
        self._done_reason = "running"
        self._last_action = None
        self._last_ack = None
        self._latest_pd_frame = None
        self._latest_osa_frame = None
        self._previous_mean_abs_error_pm = self._alignment_metrics()["mean_abs_error_pm"]
        return self._build_observation()

    @staticmethod
    def _clone_task_spec(task_spec: TaskSpec) -> TaskSpec:
        return TaskSpec(
            target_resonances_nm=np.asarray(task_spec.target_resonances_nm, dtype=float).copy(),
            tolerance_pm=float(task_spec.tolerance_pm),
            max_episode_time_ms=float(task_spec.max_episode_time_ms),
            max_control_steps=int(task_spec.max_control_steps),
            success_hold_steps=int(task_spec.success_hold_steps),
            action_budget=float(task_spec.action_budget),
            observation_budget=float(task_spec.observation_budget),
            allowed_instruments=tuple(task_spec.allowed_instruments),
        )

    def snapshot(self) -> AgentEnvSnapshot:
        if self.runtime is None:
            raise RuntimeError("reset() must be called before snapshot()")
        return AgentEnvSnapshot(
            task_spec=self._clone_task_spec(self.task_spec),
            runtime_snapshot=self.runtime.snapshot(),
            accountant_snapshot=self.accountant.snapshot_state(),
            episode_log=copy.deepcopy(self.episode_log),
            episode_start_ms=float(self._episode_start_ms),
            step_index=int(self._step_index),
            success_streak=int(self._success_streak),
            previous_mean_abs_error_pm=float(self._previous_mean_abs_error_pm),
            done=bool(self._done),
            done_reason=str(self._done_reason),
            last_action=None if self._last_action is None else copy.deepcopy(self._last_action),
            last_ack=None if self._last_ack is None else copy.deepcopy(self._last_ack),
            latest_pd_frame=None if self._latest_pd_frame is None else copy.deepcopy(self._latest_pd_frame),
            latest_osa_frame=None if self._latest_osa_frame is None else copy.deepcopy(self._latest_osa_frame),
        )

    def restore(self, snapshot: AgentEnvSnapshot) -> None:
        if self.runtime is None:
            self.runtime = self.runtime_factory()
        self.task_spec = self._clone_task_spec(snapshot.task_spec)
        self.runtime.restore(snapshot.runtime_snapshot)
        self.accountant.restore_state(snapshot.accountant_snapshot)
        self.episode_log = copy.deepcopy(snapshot.episode_log)
        self._episode_start_ms = float(snapshot.episode_start_ms)
        self._step_index = int(snapshot.step_index)
        self._success_streak = int(snapshot.success_streak)
        self._previous_mean_abs_error_pm = float(snapshot.previous_mean_abs_error_pm)
        self._done = bool(snapshot.done)
        self._done_reason = str(snapshot.done_reason)
        self._last_action = None if snapshot.last_action is None else copy.deepcopy(snapshot.last_action)
        self._last_ack = None if snapshot.last_ack is None else copy.deepcopy(snapshot.last_ack)
        self._latest_pd_frame = None if snapshot.latest_pd_frame is None else copy.deepcopy(snapshot.latest_pd_frame)
        self._latest_osa_frame = None if snapshot.latest_osa_frame is None else copy.deepcopy(snapshot.latest_osa_frame)

    def fork(self) -> "AgentEnv":
        clone = AgentEnv(
            runtime_factory=self.runtime_factory,
            task_spec=self.task_spec,
            budget_config=self.accountant.config,
        )
        clone.restore(self.snapshot())
        return clone

    def step(self, action: dict[str, Any]) -> AgentStepResult:
        if self.runtime is None:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            observation = self._build_observation()
            return AgentStepResult(observation=observation, reward=0.0, done=True, info=self.episode_summary())

        self._last_action = dict(action)
        action_type = action.get("type")
        action_cost = 0.0
        penalty = 0.0

        if action_type == "set_voltage":
            ack = self.runtime.apply_voltage(int(action["channel"]), float(action["voltage_v"]))
            self._last_ack = ack
            action_cost, penalty = self.accountant.charge_voltage(ack)
        elif action_type == "wait":
            action_cost, penalty = self.accountant.charge_wait(float(action["dt_ms"]))
            state = self.runtime.step(float(action["dt_ms"]))
            penalty += self.accountant.charge_safety(state.total_shift_warning_mask)
        elif action_type == "read_pd":
            self._ensure_instrument_allowed("PD")
            frame = self.runtime.read_pd(
                input_powers_mw=action.get("input_powers_mw"),
                wavelengths_nm=action.get("wavelengths_nm"),
            )
            self._latest_pd_frame = frame
            action_cost, penalty = self.accountant.charge_pd(frame)
            penalty += self.accountant.charge_safety(self.runtime.plant.latent_state().total_shift_warning_mask)
        elif action_type == "read_osa":
            self._ensure_instrument_allowed("OSA")
            frame = self.runtime.read_osa(
                center_nm=action.get("center_nm"),
                span_nm=action.get("span_nm"),
            )
            self._latest_osa_frame = frame
            action_cost, penalty = self.accountant.charge_osa(frame)
            penalty += self.accountant.charge_safety(self.runtime.plant.latent_state().total_shift_warning_mask)
        elif action_type == "finish":
            pass
        else:
            raise ValueError(f"unsupported action type: {action_type}")

        self._step_index += 1
        metrics = self._alignment_metrics()
        mean_abs_error_pm = metrics["mean_abs_error_pm"]
        max_abs_error_pm = metrics["max_abs_error_pm"]
        reward = (self._previous_mean_abs_error_pm - mean_abs_error_pm) - action_cost - penalty
        self._previous_mean_abs_error_pm = mean_abs_error_pm

        if max_abs_error_pm <= self.task_spec.tolerance_pm:
            self._success_streak += 1
        else:
            self._success_streak = 0

        if action_type == "finish":
            self._done = True
            if max_abs_error_pm <= self.task_spec.tolerance_pm:
                self._done_reason = "agent_finish_success"
                reward += 10.0
            else:
                self._done_reason = "agent_finish_not_converged"
                reward -= 5.0
        elif self._success_streak >= self.task_spec.success_hold_steps:
            self._done = True
            self._done_reason = "converged"
            reward += 10.0
        else:
            exceeded_reason = self.accountant.exceeded_reason(self.task_spec)
            if exceeded_reason is not None:
                self._done = True
                self._done_reason = exceeded_reason
                reward -= 5.0
            elif self.elapsed_time_ms > self.task_spec.max_episode_time_ms:
                self._done = True
                self._done_reason = "time_budget_exceeded"
                reward -= 5.0
            elif self._step_index >= self.task_spec.max_control_steps:
                self._done = True
                self._done_reason = "control_step_limit_reached"
                reward -= 5.0

        observation = self._build_observation()
        info = self.episode_summary()
        info["last_action_type"] = action_type

        self.episode_log.append(
            {
                "decision_step": self._step_index,
                "action_type": action_type,
                "time_ms": observation.time_ms,
                "mean_abs_error_pm": mean_abs_error_pm,
                "max_abs_error_pm": max_abs_error_pm,
                "reward": reward,
                "done": self._done,
                "done_reason": self._done_reason,
                "action_budget_used": observation.budget_state.action_budget_used,
                "observation_budget_used": observation.budget_state.observation_budget_used,
                "num_voltage_commands": observation.budget_state.num_voltage_commands,
                "num_pd_reads": observation.budget_state.num_pd_reads,
                "num_osa_reads": observation.budget_state.num_osa_reads,
                "num_clamped_actions": observation.budget_state.num_clamped_actions,
                "num_saturated_measurements": observation.budget_state.num_saturated_measurements,
                "num_safety_warnings": observation.budget_state.num_safety_warnings,
                "num_safety_warning_active_steps": observation.budget_state.num_safety_warning_active_steps,
            }
        )

        return AgentStepResult(observation=observation, reward=reward, done=self._done, info=info)

    @property
    def elapsed_time_ms(self) -> float:
        if self.runtime is None:
            return 0.0
        return float(self.runtime.plant.time_ms - self._episode_start_ms)

    def episode_summary(self) -> dict[str, Any]:
        metrics = self._alignment_metrics()
        budget = self.accountant.snapshot(self.task_spec)
        return {
            "success": self._done_reason in ("converged", "agent_finish_success"),
            "done": self._done,
            "done_reason": self._done_reason,
            "elapsed_time_ms": self.elapsed_time_ms,
            "final_mean_abs_error_pm": metrics["mean_abs_error_pm"],
            "final_max_abs_error_pm": metrics["max_abs_error_pm"],
            "action_budget_used": budget.action_budget_used,
            "observation_budget_used": budget.observation_budget_used,
            "total_budget_cost": budget.total_cost,
            "num_voltage_commands": budget.num_voltage_commands,
            "num_wait_actions": budget.num_wait_actions,
            "num_pd_reads": budget.num_pd_reads,
            "num_osa_reads": budget.num_osa_reads,
            "num_clamped_actions": budget.num_clamped_actions,
            "num_saturated_measurements": budget.num_saturated_measurements,
            "num_safety_warnings": budget.num_safety_warnings,
            "num_safety_warning_active_steps": budget.num_safety_warning_active_steps,
            "decision_steps": self._step_index,
        }

    def observation(self) -> AgentObservation:
        return self._build_observation()

    def _build_observation(self) -> AgentObservation:
        if self.runtime is None:
            raise RuntimeError("runtime is not initialized")
        budget_state = self.accountant.snapshot(self.task_spec)
        history_summary = {
            "num_voltage_commands": budget_state.num_voltage_commands,
            "num_wait_actions": budget_state.num_wait_actions,
            "num_pd_reads": budget_state.num_pd_reads,
            "num_osa_reads": budget_state.num_osa_reads,
            "last_pd_quality": None if self._latest_pd_frame is None else self._latest_pd_frame.quality_flag,
            "last_osa_quality": None if self._latest_osa_frame is None else self._latest_osa_frame.quality_flag,
        }
        return AgentObservation(
            time_ms=self.elapsed_time_ms,
            task_spec=self.task_spec,
            commanded_voltages_v=self.runtime.plant.target_voltages_v.copy(),
            budget_state=budget_state,
            last_action=None if self._last_action is None else dict(self._last_action),
            last_ack=self._last_ack,
            latest_pd_frame=self._latest_pd_frame,
            latest_osa_frame=self._latest_osa_frame,
            history_summary=history_summary,
        )

    def _ensure_instrument_allowed(self, instrument_type: str) -> None:
        if instrument_type not in self.task_spec.allowed_instruments:
            raise RuntimeError(f"{instrument_type} is not allowed for this task")

    def _alignment_metrics(self) -> dict[str, float]:
        if self.runtime is None:
            return {"mean_abs_error_pm": 0.0, "max_abs_error_pm": 0.0}
        state = self.runtime.plant.latent_state()
        error_pm = np.abs(state.effective_resonances_nm - self.task_spec.target_resonances_nm) * 1000.0
        return {
            "mean_abs_error_pm": float(np.mean(error_pm)),
            "max_abs_error_pm": float(np.max(error_pm)),
        }


def simulate_target_resonances(
    runtime_factory: Callable[[], SimulationRuntime],
    target_voltages_v: np.ndarray,
    settle_ms: float,
    global_temp_shift_nm: float = 0.0,
) -> np.ndarray:
    runtime = runtime_factory()
    if global_temp_shift_nm != 0.0:
        runtime.plant.set_global_temp_shift_nm(float(global_temp_shift_nm))

    target_voltages_v = np.asarray(target_voltages_v, dtype=float)
    if target_voltages_v.shape != (runtime.plant.num_rings,):
        raise ValueError("target_voltages_v must match plant num_rings")

    for channel, voltage_v in enumerate(target_voltages_v):
        runtime.plant.issue_command(channel, float(voltage_v))
    runtime.step(float(settle_ms))
    return runtime.plant.latent_state().effective_resonances_nm.copy()
