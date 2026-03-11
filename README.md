# photonic-sim

`photonic-sim` 是一个面向 **WDM MRR 在线调谐 / 校准 / 控制验证** 的研究级仿真库。  
当前版本聚焦于“论文充分仿真”的第一层目标：

- 不追求工业级多物理场高精度
- 不把所有真实细节一次堆满
- 优先保证 **闭环语义正确、观测边界正确、误差来源正确**

这版代码的核心用途是为以下研究任务提供底座：

- MRR 阵列在线重调
- calibration-aware active sensing
- budgeted observation
- collision-aware retuning
- 后续的 memory / agent / controller 策略验证

## 设计原则

这个库不是传统的 “公式演示 + 一次性 forward” 工具库。  
它也不是工业级 TCAD。

它追求的是：

**问题驱动的最小真实（minimal realism for the target research question）**

也就是只保留会改变论文结论的因素：

- 显式 latent plant state
- 动作驱动的时间推进
- 分离的仪器层
- 固定量程 ADC
- 观测时间戳 / 版本 / 质量标志
- 最小执行器约束和安全边界

## 当前版本已经具备的能力

### 1. MRR 阵列 latent plant

- `MRRArrayPlant`
- 每个 ring 有独立 actuator target / actuator state / thermal state
- 支持全局热串扰矩阵
- 支持随机漂移
- 支持全局温漂偏置

### 2. Step-based runtime

- `issue_command()` / `apply_voltage()`
- `step(dt_ms)`
- 显式时间推进，而不是瞬时写入终态

### 3. 一阶热动态

当前动力学链路为：

`voltage -> electrical power -> thermal power -> resonance shift`

这比“电压直接低通到 shift”更接近热调谐设备的基本物理语义，同时仍保持足够轻量。

### 4. Instrument layer

当前提供两类仪器：

- `PDInstrument`
- `OSAInstrument`

二者都返回统一的 `MeasurementFrame`，包含：

- `timestamp_ms`
- `calib_version`
- `quality_flag`
- `payload`
- `metadata`

### 5. Fixed-range ADC

`PDInstrument` 使用固定满量程 `full_scale_current_ma` 和固定 bitwidth：

- 不再按当前样本动态缩放
- 能显式表达量化误差
- 能显式表达 saturation

### 6. Execution and safety

当前提供最小执行器和安全模块：

- `ActionExecutor`
  - 电压 clamp
  - slew rate
- `SafetyGuard`
  - self shift clamp
  - total shift warning

## 当前版本还没有覆盖的内容

这很重要。当前版本还 **不是** 完整的 calibration-aware online retuning simulator。

仍缺少：

- calibration bootstrap
- belief state estimator
- agent / controller layer
- active probe action
- budget accountant
- add-drop / drop-port 精细模型
- 更真实的 OSA RBW 卷积
- 更复杂的热-电 RC 网络
- 多仪器异步调度总线

这些会在现有 runtime 骨架上继续迭代。

## 目录结构

```text
photonic-sim/
├── photonic_sim/
│   ├── __init__.py
│   ├── config.py
│   ├── execution.py
│   ├── instruments.py
│   ├── physics.py
│   ├── plant.py
│   ├── runtime.py
│   └── types.py
├── examples/
│   └── verify_runtime.py
├── tests/
│   └── test_runtime.py
├── MIGRATION_NOTES.md
├── README.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## 安装

### 方式 1：本地可编辑安装

```powershell
cd photonic-sim
pip install -e .
```

### 方式 2：不安装，直接在仓库目录运行

当前 `examples/verify_runtime.py` 和 `tests/test_runtime.py` 已经带了本地路径注入，
所以在仓库根目录下可以直接运行。

## 快速上手

### 1. 创建一个最小 MRR runtime

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

### 3. 读取 PD 观测

```python
frame = runtime.read_pd(input_powers_mw=np.ones(3))

print(frame.instrument_type)
print(frame.timestamp_ms)
print(frame.quality_flag)
print(frame.payload["quantized_currents_ma"])
print(frame.metadata)
```

### 4. 读取 OSA 观测

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

## 如何验证当前库

### 1. 运行单元测试

```powershell
cd photonic-sim
python -m pytest tests -q
```

当前测试覆盖：

- actuator / thermal state 需要通过时间推进才生效
- executor clamp 和 safety warning
- fixed-range ADC saturation
- fresh / stale measurement frame 语义

### 2. 运行手动验证脚本

```powershell
python examples/verify_runtime.py
```

该脚本会打印 3 类验证结果：

- `voltage -> power -> thermal state -> shift`
- `PD / OSA` measurement frame
- executor clamp 与 safety guard

## 当前适合做哪些实验

在当前版本上，已经适合做以下实验：

- actuator step response
- thermal crosstalk sweep
- drift sensitivity sweep
- PD / OSA 观测链测试
- ADC saturation / bitwidth sweep
- safety threshold stress test
- calibration 数据采集前的基础动态验证

当前还不适合直接做：

- 完整 calibration-aware retuning 闭环论文主实验
- probe-driven identity recovery
- memory / agent / MPC 公平对比

## 和旧版相比最大的变化

旧版主要问题是：

- 目标权重直接反解
- 无显式时间推进
- 光域 / 电域混在一个 carrier
- ADC 按当前样本动态缩放

新版重构优先修正了这些结构问题。详细说明见：

- [MIGRATION_NOTES.md](./MIGRATION_NOTES.md)

## 下一步路线

建议接下来的实现顺序：

1. `CalibrationBootstrap`
2. `BeliefStateEstimator`
3. `Budget / Cost Accountant`
4. `Agent / Controller Core`

也就是说，当前版本已经把“器件-执行器-仪器-runtime”这层打通，
后续可以在它之上长出校准和智能控制层。
