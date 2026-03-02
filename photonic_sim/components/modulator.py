"""
电光调制器 (EOM)

将电域数据调制到 WDM 各波长通道上。
在微梳 PPU 架构中，EOM 负责将图像像素值调制为光强度。
"""
import numpy as np
from .base import OpticalComponent, OpticalSignal


class EOMModulator(OpticalComponent):
    """
    电光调制器

    将输入数据调制到光信号的各通道上。
    输出功率 = 输入功率 × |data| (强度调制)

    Attributes:
        insertion_loss_db: 调制器插入损耗 [dB]
        bandwidth_ghz:     调制带宽 [GHz]
    """

    def __init__(self, insertion_loss_db: float = 3.0,
                 bandwidth_ghz: float = 40.0):
        self.insertion_loss_db = insertion_loss_db
        self.bandwidth_ghz = bandwidth_ghz
        self._loss_linear = 10 ** (-insertion_loss_db / 10)

    @property
    def name(self) -> str:
        return f"EOM(IL={self.insertion_loss_db}dB)"

    def forward(self, signal: OpticalSignal) -> OpticalSignal:
        """
        调制操作

        如果信号携带 data，按 data 值调制各通道功率；
        否则仅施加插入损耗。
        """
        out = signal.copy()
        out.powers = out.powers * self._loss_linear

        if out.data is not None:
            # 强度调制: P_out = P_in × |data|
            out.powers = out.powers * np.abs(out.data[:out.num_channels])

        return out

    def load_data(self, signal: OpticalSignal,
                  data: np.ndarray) -> OpticalSignal:
        """
        便捷方法：加载数据到信号并执行调制

        Args:
            signal: 输入光信号
            data:   要调制的数据, shape=(N,) 或匹配通道数

        Returns:
            调制后的光信号
        """
        out = signal.copy()
        out.data = np.asarray(data, dtype=float).flatten()
        return self.forward(out)
