# photonic-sim 设计文档：可扩展 WDM MRR 仿真平台

## 定位

这是一个**底层光子仿真组件库**，不是应用层项目。
- ✅ 提供：MRR 建模、WDM 通道管理、光信号传输链路、物理效应仿真
- ❌ 不包含：PPU 卷积、MNIST 推理、Agent 控制等应用逻辑
- 上游项目（如 optical-agent）通过 `from photonic_sim import ...` 引用本库

## 架构

```
photonic_sim/
├── core/                      # 物理基础层
│   ├── physics.py             # [已完成] 物理常量 + 核心公式
│   ├── wavelength_grid.py     # [已完成] 微梳/激光器波长管理
│   ├── mrr.py                 # [已完成] 单环 MRR 模型 (V²调谐, Lorentzian)
│   └── mrr_bank.py            # [已完成] 权重阵列 (N×N 串扰矩阵)
│
├── components/                # 可扩展光学组件层
│   ├── base.py                # [新建] 光学组件抽象基类
│   ├── signal.py              # [新建] 多波长光信号表示
│   ├── modulator.py           # [新建] EOM 电光调制器
│   ├── photodetector.py       # [新建] 光电探测器 (PD)
│   └── fiber.py               # [新建] 光纤传输 (损耗 + SRS)
│
├── link/                      # 光链路编排层
│   └── optical_link.py        # [新建] 组件串联 → 端到端链路仿真
│
└── utils/                     # 工具
    └── visualization.py       # [新建] 频谱绘图工具
```

## 核心设计：组件模型 (Component Model)

### 抽象基类

所有光学组件继承 `OpticalComponent`，实现统一接口：

```python
class OpticalComponent(ABC):
    """光学组件抽象基类"""

    @abstractmethod
    def forward(self, signal: OpticalSignal) -> OpticalSignal:
        """处理输入光信号，返回输出光信号"""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...
```

### 光信号表示

```python
@dataclass
class OpticalSignal:
    """多波长光信号"""
    wavelengths: np.ndarray   # 各通道波长 [nm], shape=(N,)
    powers: np.ndarray        # 各通道功率 [mW], shape=(N,)
    data: np.ndarray = None   # 各通道承载数据 (可选), shape=(N,) or (N,M)
```

### 可扩展组件

| 组件 | `forward()` 行为 | 物理公式 |
|------|-----------------|---------|
| `MRRBankFilter` | 按权重阵列对各通道功率加权 | Lorentzian 透射 |
| `EOMModulator` | 将电域数据调制到各波长通道 | `signal.data = input_data` |
| `Photodetector` | WDM 通道信号光电转换 + 求和 | `I = R × P + noise` |
| `FiberSpan` | 功率衰减 + SRS 倾斜 | `P_out = P_in × 10^(-αL/10)` |

### 光链路编排

```python
class OpticalLink:
    """将多个组件串联为端到端链路"""

    def __init__(self, components: list[OpticalComponent]):
        self.components = components

    def forward(self, signal: OpticalSignal) -> OpticalSignal:
        for comp in self.components:
            signal = comp.forward(signal)
        return signal
```

上游 PPU 的使用方式变为：
```python
from photonic_sim.core import WavelengthGrid, MRRWeightBank
from photonic_sim.components import EOMModulator, Photodetector, MRRBankFilter
from photonic_sim.link import OpticalLink

# 搭建 PPU 信号链路
link = OpticalLink([
    EOMModulator(data=image_patch),
    MRRBankFilter(weight_bank=bank),
    Photodetector(responsivity=0.8),
])
output = link.forward(comb_signal)
```

## 已完成模块 (core/)

| 文件 | 状态 | 内容 |
|------|------|------|
| `physics.py` | ✅ | PhysicsConstants, lorentzian_transmission, inverse_lorentzian, voltage_to_shift, shift_to_voltage |
| `wavelength_grid.py` | ✅ | WavelengthGrid (微梳波长生成) |
| `mrr.py` | ✅ | MRR (V² 调谐, Lorentzian 透射, FSR 折叠, 权重映射) |
| `mrr_bank.py` | ✅ | MRRWeightBank (N×N 串扰矩阵, 向量化物理引擎, 随机漂移) |

## 待实现模块

### Phase 1: 组件层 (`components/`)

1. **`base.py`** — `OpticalComponent` 抽象基类 + `OpticalSignal` 数据类
2. **`signal.py`** — 光信号创建辅助（从 WavelengthGrid 生成均匀功率梳信号）
3. **`modulator.py`** — EOM 调制器（将数据加载到各通道）
4. **`photodetector.py`** — PD（含散粒噪声 + ADC 量化）
5. **`fiber.py`** — 光纤（衰减 + 简化 SRS）

### Phase 2: 链路层 (`link/`)

6. **`optical_link.py`** — 组件串联编排器

### Phase 3: 工具 (`utils/`)

7. **`visualization.py`** — 频谱绘图（MRR 透射谱、链路功率分布）

### Phase 4: 测试 (`tests/`)

8. **`test_mrr.py`** — 物理正确性验证（7 项测试）
9. **`test_link.py`** — 端到端链路验证

## 验证计划

```bash
cd E:\Antigravity\projects\photonic-sim
pip install -e .
python -m pytest tests/ -v
```

测试项:
1. Lorentzian 谱形: 谐振处 T=T_min, 远离谐振 T→1.0
2. V² 调谐: 电压翻倍 → 偏移 4 倍
3. FSR 周期性: 偏移 1 FSR 后透过率回到同一值
4. 串扰矩阵: C[i,i]=1, C 对称, 最近邻~6-8%, 远距离<1%
5. 权重编程: 无漂移时, 误差 < 0.01
6. 光链路: EOM→MRR→PD 链路端到端通过
