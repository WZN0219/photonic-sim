# Migration Notes

这个文件用于把原始 `photonic-sim-main` 的关键问题，映射到当前新版 `photonic-sim` 已经完成或尚未完成的重构点上。

## 原始库中的关键结构问题

## 1. 没有显式时间推进

原始问题：

- `OpticalLink.forward()` 只是一次同步前向。
- `MRRWeightBank.program_weights()` 会瞬时写入最终状态。
- 没有 `dt`、没有 actuator settling、没有 runtime step。

当前处理：

- 新增 `MRRArrayPlant.issue_command()`
- 新增 `MRRArrayPlant.step(dt_ms)`
- actuator 电压和谐振漂移通过时间推进演化
- 进一步细化为：
  - `voltage command`
  - `command power`
  - `thermal power state`
  - `shift`

## 2. 控制接口是目标反解，不是动作驱动

原始问题：

- 目标权重会直接反解为理想 shift。
- heater voltage 只是展示值，不是真正驱动状态变化的因果量。

当前处理：

- 新底座把 `target_voltage` 作为真正控制输入
- `actuator_voltages_v` 和 `target_voltages_v` 分开建模
- 物理状态由 `step()` 驱动更新
- 新增 `ActionExecutor` 对命令进行 clamp / reject
- 新增 `SafetyGuard` 对热状态和总 shift 做最小安全检查

## 3. 观测层和物理层没有分开

原始问题：

- `OpticalSignal.powers` 在 PD 后被覆写成电流
- 光域和电域共用一个 carrier
- measurement frame 不带 timestamp / calib_version

当前处理：

- 新增 `MeasurementFrame`
- OSA / PD 返回带 `timestamp_ms`、`calib_version`、`quality_flag` 的结构化观测
- measurement payload 和 latent state 语义拆开

## 4. ADC 量化会按当前样本动态缩放

原始问题：

- 旧版 PD 用当前样本自己的最大值决定量化步长
- 无法真实表达固定量程、饱和和动态范围不足

当前处理：

- 新版 PD 使用固定 `full_scale_current_ma`
- 明确返回 `adc_lsb_ma`
- 明确报告 `saturated`

## 5. 原始测试没有覆盖 runtime 语义

原始问题：

- 旧测试主要检查公式局部正确
- 没有验证 step-based runtime、fresh/stale frame、固定量程 ADC

当前处理：

- 新增 `tests/test_runtime.py`
- 覆盖：
  - command 必须经时间推进才生效
  - 固定量程 ADC
  - timestamped frame + stale/fresh 观测

## 当前仍未完成的重构项

- add-drop / drop-port 更真实的端口模型
- calibration bootstrap 模块
- controller / agent 接口
- OSA RBW 卷积与更真实噪声
- 事件调度器和多仪器异步采样总线
- 更真实的热-电动态与局部温度状态
- 安全约束、slew rate 和 actuator error code

## 推荐后续顺序

1. 先补 `ActionExecutor + SafetyGuard`
2. 再补 `CalibrationBootstrap`
3. 再定义 `BeliefStateEstimator` 的输入输出
4. 最后再接 `Agent Core`
