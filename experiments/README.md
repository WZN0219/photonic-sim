# Experiments

本目录放的是当前 `photonic-sim` 第一阶段最适合直接开展的实验脚本。

这些实验的目标不是直接产出最终闭环控制结果，而是先验证：

- 执行器动态是否可辨识
- 热串扰是否可量化
- PD / OSA 观测链是否足以支撑后续校准
- ADC 设置会不会改变结论
- 漂移数据是否足够支持后续 `CalibrationBootstrap`

## 当前实验脚本

- `run_step_response.py`
  - 单通道执行器阶跃响应
  - 输出时间序列 CSV
- `run_crosstalk_scan.py`
  - 单通道驱动、全阵列串扰扫描
  - 输出宽表 CSV
- `run_observation_chain_sweep.py`
  - PD ADC sweep
  - OSA 采样参数 sweep
  - 输出两个 CSV
- `run_drift_observation_dataset.py`
  - 长时间漂移场景观测数据生成
  - 输出 latent / PD / OSA 三类 CSV
- `run_calibration_bootstrap.py`
  - 从前面四类实验输出中提取最小校准摘要
  - 输出 `calibration_bootstrap.json`
- `run_agent_retuning_baseline.py`
  - 在统一 `TaskSpec + AgentEnv + BudgetConfig` 下运行第一版 bootstrap-guided baseline
  - 输出 `summary.csv`、`task.csv`、`trajectory.csv`
- `run_belief_recovery_probe.py`
  - 运行一条最小 belief update / stale OSA / recovery trigger 探针链路
  - 输出 `belief_probe.csv`

## 推荐执行顺序

1. `run_step_response.py`
2. `run_crosstalk_scan.py`
3. `run_observation_chain_sweep.py`
4. `run_drift_observation_dataset.py`
5. `run_calibration_bootstrap.py`
6. `run_agent_retuning_baseline.py`
7. `run_belief_recovery_probe.py`

## 运行示例

```powershell
python experiments/run_step_response.py
python experiments/run_crosstalk_scan.py
python experiments/run_observation_chain_sweep.py
python experiments/run_drift_observation_dataset.py
python experiments/run_calibration_bootstrap.py
python experiments/run_agent_retuning_baseline.py
python experiments/run_belief_recovery_probe.py
```

所有输出默认写到：

```text
experiments/outputs/
```

## 可视化报告

在四个实验脚本产出 CSV 后，可以再运行：

```powershell
python experiments/visualize_experiments.py
```

默认会生成：

```text
experiments/outputs/experiment_report.html
```

这份 HTML 会把四个 `run_*.py` 分别对应的测试目的、关键指标和样例图放到同一页里，方便快速浏览。

## Agent Baseline

`run_agent_retuning_baseline.py` 是当前从“前置表征实验”迈向“agent 调控实验”的第一步。

默认会：

- 读取 `baseline_20260311` 里的 `CalibrationBootstrap`
- 构造一个温和偏差的 `mild` retuning task
- 运行 bootstrap-guided baseline controller
- 输出：
  - `summary.csv`
  - `task.csv`
  - `trajectory.csv`

如果要做 stress test，可以切到：

```powershell
python experiments/run_agent_retuning_baseline.py --scenario hard
```

## Belief / Recovery Probe

`run_belief_recovery_probe.py` 用来验证当前最小推理原语是否已经闭合：

- `CalibrationState`
- `BeliefState`
- `SimpleBeliefStateEstimator`
- `RecoveryTrigger`

默认会依次产生：

- fresh OSA belief update
- stale OSA detection
- recovery trigger decision

输出文件为：

```text
experiments/outputs/belief_recovery_probe/belief_probe.csv
```
