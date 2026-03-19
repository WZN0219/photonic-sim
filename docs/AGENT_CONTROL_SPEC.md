# Runtime Interface Spec

这份文档描述 `photonic-sim` 当前提供的底层运行时接口与语义约定。

它关注的是：

- 仿真任务如何定义
- runtime 支持哪些动作
- 观测对象包含什么信息
- 成本、预算、成功和失败如何计算

它不讨论上层策略、记忆系统或复杂决策结构。

## 1. TaskSpec

每个 episode 由 `TaskSpec` 定义，当前包含：

- `target_resonances_nm`
- `tolerance_pm`
- `max_episode_time_ms`
- `max_control_steps`
- `success_hold_steps`
- `action_budget`
- `observation_budget`
- `allowed_instruments`

这些字段决定了：

- 调谐目标
- 收敛容差
- 时间与控制步数上限
- 动作与观测预算
- 允许调用的仪器

## 2. 动作接口

当前 runtime 支持的离散动作类型：

- `set_voltage(channel, voltage_v)`
- `wait(dt_ms)`
- `read_pd(input_powers_mw=None, wavelengths_nm=None)`
- `read_osa(center_nm=None, span_nm=None)`
- `finish()`

说明：

- `set_voltage` 修改目标电压，不直接写入终态
- `wait` 推进热动态
- `read_pd` / `read_osa` 是主动观测动作
- `finish` 用于显式结束当前 episode

## 3. Runtime 语义

核心入口：

- `SimulationRuntime.apply_voltage()`
- `SimulationRuntime.step()`
- `SimulationRuntime.read_pd()`
- `SimulationRuntime.read_osa()`

当前核心演化链路：

`voltage -> electrical power -> thermal power -> resonance shift`

其中：

- 执行器变化受 `slew_rate_v_per_ms` 限制
- 热状态按一阶动态更新
- 共振偏移由 tuning efficiency、crosstalk、漂移和全局温漂共同决定

## 4. 仪器观测对象

`PDInstrument` 和 `OSAInstrument` 都返回统一的 `MeasurementFrame`：

- `instrument_type`
- `timestamp_ms`
- `calib_version`
- `quality_flag`
- `payload`
- `metadata`

当前约定：

- `quality_flag` 为 `fresh` 或 `stale`
- stale frame 会保留上一帧的关键 metadata，并额外带上：
  - `source_timestamp_ms`
  - `frame_period_ms`

## 5. AgentObservation

`AgentEnv` 每一步返回 `AgentObservation`，当前包含：

- `time_ms`
- `task_spec`
- `commanded_voltages_v`
- `budget_state`
- `last_action`
- `last_ack`
- `latest_pd_frame`
- `latest_osa_frame`
- `history_summary`

这层对象的作用是：

- 汇总当前运行时状态
- 暴露最近一次观测
- 提供预算和历史摘要

## 6. 预算与成本

`BudgetConfig` 当前定义：

- `voltage_action_cost`
- `wait_cost_per_ms`
- `pd_read_cost`
- `osa_read_cost`
- `clamp_penalty_cost`
- `saturation_penalty_cost`
- `safety_warning_cost`

`BudgetAccountant` 当前跟踪：

- `action_budget_used`
- `observation_budget_used`
- `num_voltage_commands`
- `num_wait_actions`
- `num_pd_reads`
- `num_osa_reads`
- `num_clamped_actions`
- `num_saturated_measurements`
- `num_safety_warnings`
- `num_safety_warning_active_steps`

## 7. 成功与失败

当前 `AgentEnv` 的成功 / 失败语义：

成功：

- `max_abs_error_pm <= tolerance_pm`
- 且连续满足 `success_hold_steps`
- 或者调用 `finish()` 时已在容差内

失败：

- `action_budget_exceeded`
- `observation_budget_exceeded`
- `time_budget_exceeded`
- `control_step_limit_reached`
- `finish()` 时仍未收敛

## 8. 校准与推理原语

当前提供：

- `CalibrationBootstrap`
- `CalibrationState`
- `BeliefState`
- `SimpleBeliefStateEstimator`
- `RecoveryTrigger`
- `BootstrapRetuningController`

这些模块的用途是：

- 从基础实验输出构建最小校准摘要
- 维护最小状态估计
- 触发恢复建议
- 运行一个可复现的 baseline 闭环

## 9. Snapshot / Restore / Fork

当前支持以下状态接口：

- `MRRArrayPlant.snapshot() / restore() / fork()`
- `SimulationRuntime.snapshot() / restore() / fork()`
- `AgentEnv.snapshot() / restore() / fork()`

这些接口用于：

- 复制当前仿真状态
- 构造可重复的分支 rollout
- 做一致性测试与调试

当前约定是：

- restore 后的状态应与 snapshot 对应状态保持一致
- fork 后继续执行同一动作序列，应得到一致结果

## 10. 非目标

当前文档不覆盖：

- 上层任务规划
- 长期记忆
- 复杂策略搜索
- 大规模实验编排
- 高层决策框架

这些属于上层项目，不属于 `photonic-sim` 这层底层仿真库。
