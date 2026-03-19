# photonic-sim

`photonic-sim` 是一个面向 **WDM MRR 阵列** 的底层仿真库，用于提供：

- MRR 阵列植物理状态演化
- 电压动作与时间推进
- `PD` / `OSA` 仪器观测
- 基础校准摘要与最小推理原语
- 可复现的 characterization / baseline 实验脚本

## 主要能力

### 1. 植物理模型

- `MRRArrayPlant`
- 每个 ring 的独立电压目标、执行器状态、热状态
- 一阶热动态
- 全局热串扰矩阵
- 全局温漂偏置
- 随机漂移项

当前核心链路为：

`voltage -> electrical power -> thermal power -> resonance shift`

### 2. Runtime

- `SimulationRuntime.apply_voltage()`
- `SimulationRuntime.step(dt_ms)`
- `SimulationRuntime.read_pd()`
- `SimulationRuntime.read_osa()`

支持显式时间推进，而不是一步写入终态。

### 3. 仪器层

当前提供两类仪器：

- `PDInstrument`
- `OSAInstrument`

统一返回 `MeasurementFrame`，包含：

- `timestamp_ms`
- `calib_version`
- `quality_flag`
- `payload`
- `metadata`

### 4. 执行器与安全约束

- `ActionExecutor`
  - 电压 clamp
  - slew rate
- `SafetyGuard`
  - self shift clamp
  - total shift warning

### 5. 校准与推理原语

- `CalibrationBootstrap`
- `CalibrationState`
- `BeliefState`
- `SimpleBeliefStateEstimator`
- `RecoveryTrigger`
- `BootstrapRetuningController`

这些模块用于支持基础校准、状态估计和 baseline 验证。

## 安装

### 方式 1：本地可编辑安装

```powershell
cd photonic-sim
pip install -e .
```

### 方式 2：直接在仓库目录运行

当前仓库里的示例、测试和实验脚本都支持从仓库根目录直接运行。

## 快速上手

### 1. 创建一个最小 runtime

```python
import numpy as np

from photonic_sim import (
    ActionExecutorConfig,
    MRRArrayPlant,
    MRRPlantConfig,
    PDInstrument,
    PDInstrumentConfig,
    SimulationRuntime,
    build_comb_wavelengths,
)

comb = build_comb_wavelengths(center_nm=1550.0, fsr_nm=0.73, num_lines=3)

plant = MRRArrayPlant(
    num_rings=3,
    comb_wavelengths_nm=comb,
    config=MRRPlantConfig(
        thermal_tau_ms=5.0,
        drift_sigma_nm_per_s=0.0,
    ),
    rng=np.random.default_rng(0),
    action_config=ActionExecutorConfig(
        max_voltage_v=5.0,
        slew_rate_v_per_ms=2.0,
    ),
)

runtime = SimulationRuntime(
    plant=plant,
    pd_instrument=PDInstrument(
        PDInstrumentConfig(
            adc_bits=10,
            full_scale_current_ma=0.2,
            frame_period_ms=2.0,
            noise_sigma=0.001,
        ),
        rng=np.random.default_rng(1),
    ),
)
```

### 2. 下发动作并推进时间

```python
ack = runtime.apply_voltage(channel=1, voltage_v=3.0)
print(ack)

runtime.step(2.0)
runtime.step(2.0)

state = runtime.plant.latent_state()
print(state.actuator_voltages_v)
print(state.thermal_powers_mw)
print(state.effective_resonances_nm)
```

### 3. 读取 PD

```python
frame = runtime.read_pd(input_powers_mw=np.ones(3))

print(frame.instrument_type)
print(frame.timestamp_ms)
print(frame.quality_flag)
print(frame.payload["quantized_currents_ma"])
print(frame.metadata)
```

### 4. 读取 OSA

```python
from photonic_sim import OSAInstrument, OSAInstrumentConfig

runtime.osa = OSAInstrument(
    OSAInstrumentConfig(
        step_pm=10.0,
        span_nm=0.6,
        frame_period_ms=5.0,
    ),
    rng=np.random.default_rng(2),
)

osa = runtime.read_osa()
print(osa.payload["wavelengths_nm"][:5])
print(osa.payload["spectrum_dbm"][:5])
```

## 可视化示例

如果你希望先看图，再看代码，可以直接运行：

```powershell
python examples/plot_mrr_visual_overview.py
```

默认会在下面生成 4 张 PNG：

```text
examples/outputs/mrr_visual_overview/
├── resonance_positions.png
├── through_spectrum.png
├── crosstalk_matrix.png
└── crosstalk_profile.png
```

这些图分别展示：

- MRR 阵列的 base resonance 和 effective resonance
- 阵列 through 光谱
- 串扰矩阵热图
- 单个 ring 向周围 ring 的串扰衰减分布

## 如何验证

### 1. 运行测试

```powershell
cd photonic-sim
python -m pytest tests -q
```

当前测试覆盖：

- actuator / thermal state 需要通过时间推进才生效
- executor clamp 与 safety warning
- stale / fresh frame 语义
- snapshot / restore / fork 一致性
- calibration bootstrap 摘要构建
- belief update 与 recovery trigger
- baseline controller 基本闭环

### 2. 运行手动验证脚本

```powershell
python examples/verify_runtime.py
```

该脚本会打印：

- 电压到热状态与共振偏移的变化
- `PD` / `OSA` 的观测帧内容
- clamp 与 safety guard 的行为

## 实验脚本

推荐的基础实验入口：

```powershell
python experiments/run_step_response.py
python experiments/run_crosstalk_scan.py
python experiments/run_observation_chain_sweep.py
python experiments/run_drift_observation_dataset.py
python experiments/run_calibration_bootstrap.py
python experiments/run_agent_retuning_baseline.py
python experiments/run_belief_recovery_probe.py
```

这些脚本分别对应：

- 执行器阶跃响应
- 热串扰扫描
- `PD` / `OSA` 观测链 sweep
- 漂移数据集生成
- 校准摘要构建
- baseline 闭环验证
- belief / recovery 行为探针

默认输出目录：

```text
experiments/outputs/
```

## 目录结构

```text
photonic-sim/
├── photonic_sim/
│   ├── __init__.py
│   ├── agent.py
│   ├── calibration.py
│   ├── config.py
│   ├── controller.py
│   ├── execution.py
│   ├── inference.py
│   ├── instruments.py
│   ├── physics.py
│   ├── plant.py
│   ├── runtime.py
│   └── types.py
├── docs/
├── examples/
├── experiments/
├── tests/
├── MIGRATION_NOTES.md
├── README.md
├── pyproject.toml
└── requirements.txt
```

## 当前边界

当前仓库主要覆盖：

- 底层植物理与仪器语义
- 时间推进与预算语义
- 校准摘要与最小推理原语
- characterization / baseline 实验

当前没有覆盖的内容包括：

- 更复杂的多物理场模型
- 更精细的 OSA 光谱仪器建模
- 更复杂的热-电网络
- 上层任务规划、记忆系统和复杂决策策略

## 相关文件

- 迁移说明：[`MIGRATION_NOTES.md`](./MIGRATION_NOTES.md)
- 详细实验说明：[`experiments/README.md`](./experiments/README.md)
