# Agent Control Spec

这份文档用于把 `photonic-sim` 的主线重新收敛到 **agent 调控任务定义**。

当前项目已经完成 runtime、仪器层、基础实验和最小 `CalibrationBootstrap`。
下一阶段的重点不再是继续做底层架构扩张，而是明确：

- agent 的任务到底是什么
- agent 能看到什么
- agent 能做什么动作
- 成功 / 失败 / 成本如何定义
- 哪些实验是在为 agent 主线服务

## 1. 主问题定义

项目主问题定义为：

**在存在热惯性、串扰、漂移、观测延迟和测量预算限制的条件下，agent 如何通过有限动作与有限观测，把 MRR 阵列调回目标工作点。**

这里的关键不是继续追求更复杂的 plant realism，
而是建立一个对 agent 公平、清晰、可比较的调控任务。

## 2. 任务定义

每个 episode 应由 `TaskSpec` 明确指定以下内容：

- `target_resonances_nm` 或 `target_detuning_nm`
- `tolerance_pm`
- `max_episode_time_ms`
- `max_control_steps`
- `observation_budget`
- `action_budget`
- `allowed_instruments`
- `success_rule`
- `failure_rule`

推荐的默认主任务：

- 给定目标 comb line
- 允许 agent 通过电压动作和仪器观测逐步逼近
- 在预算内完成全阵列 retuning

推荐 success 条件：

- 所有 ring 的目标误差都落入容差内
- 且连续若干步保持稳定
- 且没有触发不可恢复的安全失败

推荐 failure 条件：

- 超过时间预算
- 超过动作预算
- 超过观测预算
- 出现持续 safety violation
- 达到 episode 终点仍未满足容差

## 3. 信息边界

agent **不应直接访问** `LatentPlantState`。

`LatentPlantState` 的作用应限定为：

- 仿真器内部演化
- 离线评估真值
- 调试和 oracle baseline

agent 允许看到的内容应限制为：

- `ActionAck`
- `MeasurementFrame`
- 当前时间
- 预算余量
- task spec
- 由环境汇总的历史摘要

这条边界非常重要。  
如果让 agent 直接读取 `effective_resonances_nm`、`thermal_powers_mw` 或 `drift_nm`，
那得到的就不是调控任务，而是 oracle 控制任务。

## 4. 最小动作集

建议第一阶段固定为离散语义动作，而不是一开始就追求复杂连续控制接口。

最小动作集：

- `set_voltage(channel, voltage_v)`
- `wait(dt_ms)`
- `read_pd(input_powers_mw=None, wavelengths_nm=None)`
- `read_osa(center_nm=None, span_nm=None)`
- `finish()`

说明：

- `set_voltage` 用于显式控制 ring
- `wait` 用于让热动态真正演化，而不是把时间推进隐式塞进别的动作
- `read_pd` 和 `read_osa` 必须作为有成本的主动观测动作
- `finish` 用于让 agent 显式声明“我认为已经调好了”

暂不建议第一阶段加入：

- 联合多通道大动作
- 直接设置目标 shift
- 直接访问 latent probe
- 高级脚本动作宏

## 5. 观测契约

建议定义统一的 `AgentObservation`，至少包含：

- `time_ms`
- `task_spec`
- `last_action`
- `last_ack`
- `latest_pd_frame`
- `latest_osa_frame`
- `budget_state`
- `history_summary`

其中每个测量帧必须保留：

- `timestamp_ms`
- `quality_flag`
- `calib_version`
- `payload`
- `metadata`

agent 需要显式感知：

- frame 是 `fresh` 还是 `stale`
- 该 frame 对应的真实采样时间
- ADC 是否 saturation
- OSA span / step / center 等上下文

环境不应偷做过强的信息压缩。  
但为了支持 baseline，可以额外提供轻量摘要特征，例如：

- 最近一次 PD 非零通道比例
- 最近一次 OSA 峰值位置估计
- 最近若干动作后的观测时间差

## 6. 成本与奖励

主奖励不应只看最终误差，还应同时反映观测和控制成本。

建议把目标拆成：

- `alignment_reward`
- `time_penalty`
- `action_penalty`
- `observation_penalty`
- `safety_penalty`
- `termination_bonus`

推荐惩罚项来源：

- 每次 `set_voltage`
- 每次 `wait`
- 每次 `read_pd`
- 每次 `read_osa`
- 命令被 `clamped`
- 观测发生 saturation
- 触发 total shift warning

推荐第一阶段主要优化目标：

- 最终对准误差最小
- 成功率最高
- OSA 调用次数尽量少
- 总耗时尽量短

## 7. 评估指标

所有 agent / controller 都应至少报告：

- `success_rate`
- `final_max_abs_error_pm`
- `final_mean_abs_error_pm`
- `settling_time_ms`
- `num_voltage_commands`
- `num_pd_reads`
- `num_osa_reads`
- `total_budget_cost`
- `num_clamped_actions`
- `num_safety_warnings`
- `num_saturated_measurements`

推荐测试场景分层：

- `static_offset`
- `crosstalk_dominant`
- `drift_tracking`
- `partial_observation`
- `budget_limited`

当前 experiments 目录里的基础实验主要服务于这些评估场景的参数设定，
而不是直接替代 agent benchmark。

## 8. 现有代码与主线的映射

当前已经具备：

- `SimulationRuntime.apply_voltage()`
- `SimulationRuntime.step()`
- `SimulationRuntime.read_pd()`
- `SimulationRuntime.read_osa()`
- `ActionAck`
- `MeasurementFrame`
- `CalibrationBootstrap`

这些能力足以支撑第一版 agent 环境。

当前仍缺：

- `TaskSpec`
- `BudgetAccountant`
- `AgentObservation`
- `AgentEnv`
- `BeliefStateEstimator`
- `HeuristicController`
- `Agent Core`

## 9. 非目标

在 agent 规格稳定前，以下内容不应继续成为主线：

- 进一步增加 plant 物理复杂度
- 更复杂的 OSA RBW 细节
- 更复杂的热-电网络
- 过早做 agent / MPC 公平对比
- 直接围绕 latent oracle 做控制器

这些内容以后可能有价值，
但现在都不比“明确 agent 任务定义”更重要。

## 10. 推荐执行顺序

建议把实现顺序固定为：

1. `TaskSpec`
2. `BudgetAccountant`
3. `AgentObservation`
4. `AgentEnv`
5. `BeliefStateEstimator`
6. `HeuristicController`
7. `Learned Agent`

其中第一阶段的完成标准应是：

- 可以用统一环境接口跑一个 episode
- 可以在不读取 latent state 的前提下执行脚本化 baseline
- 可以在至少一个 drift / budget 场景下输出可比较的评估指标

## 11. 当前阶段完成标准

当以下条件满足时，说明项目已经真正回到 agent 主线：

- 所有新实验都能映射到某个 `TaskSpec`
- 所有控制算法都走同一个 `AgentEnv`
- 所有比较都使用同一套 reward / budget / success 定义
- `CalibrationBootstrap` 只作为初始化先验，而不承担在线控制职责

到那时，再继续扩展 estimator、controller 或 learned agent，才是顺着主线前进。
