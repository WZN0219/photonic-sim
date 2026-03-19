# Experiments

本目录包含 `photonic-sim` 自带的基础实验脚本。

这些脚本主要用于：

- 生成 characterization 数据
- 检查 runtime / instrument / budget 语义
- 构建最小校准摘要
- 运行 baseline 验证

默认输出目录：

```text
experiments/outputs/
```

## 脚本列表

### `run_step_response.py`

用途：

- 单通道执行器阶跃响应
- 记录电压、功率、热状态和共振偏移的时间序列

主要输出：

- `step_response.csv`

### `run_crosstalk_scan.py`

用途：

- 单通道驱动下的全阵列串扰扫描

主要输出：

- `crosstalk_scan.csv`

### `run_observation_chain_sweep.py`

用途：

- `PD` ADC 参数 sweep
- `OSA` 采样参数 sweep
- 检查 stale / fresh frame 行为和局部峰位误差

主要输出：

- `observation_chain/pd_sweep.csv`
- `observation_chain/osa_sweep.csv`

### `run_drift_observation_dataset.py`

用途：

- 生成长时间漂移场景下的 latent / `PD` / `OSA` 数据集

主要输出：

- `drift_dataset/latent_state.csv`
- `drift_dataset/pd_frames.csv`
- `drift_dataset/osa_frames.csv`

### `run_calibration_bootstrap.py`

用途：

- 从前置实验输出中构建最小校准摘要

主要输出：

- `calibration_bootstrap.json`

### `run_agent_retuning_baseline.py`

用途：

- 在统一 `TaskSpec + AgentEnv + BudgetConfig` 下运行 bootstrap-guided baseline

主要输出：

- `summary.csv`
- `task.csv`
- `trajectory.csv`

### `run_belief_recovery_probe.py`

用途：

- 检查 `CalibrationState`
- 检查 `BeliefState`
- 检查 `SimpleBeliefStateEstimator`
- 检查 `RecoveryTrigger`

主要输出：

- `belief_probe.csv`

### `visualize_experiments.py`

用途：

- 把前置实验输出整理成 HTML 报告

主要输出：

- `experiment_report.html`

## 推荐执行顺序

如果要从头生成一套基础实验结果，建议按这个顺序运行：

```powershell
python experiments/run_step_response.py
python experiments/run_crosstalk_scan.py
python experiments/run_observation_chain_sweep.py
python experiments/run_drift_observation_dataset.py
python experiments/run_calibration_bootstrap.py
python experiments/run_agent_retuning_baseline.py
python experiments/run_belief_recovery_probe.py
```

如果前四类 CSV 已经生成，再运行：

```powershell
python experiments/visualize_experiments.py
```

## 说明

本目录中的脚本主要服务于底层仿真库验证、参数扫描和 baseline 检查。

更高层的任务设计、复杂策略实验或上层项目逻辑，不属于本目录职责。
