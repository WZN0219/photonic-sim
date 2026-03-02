"""
MRR Weight Bank 作为光学组件

将 core.MRRWeightBank 包装为 OpticalComponent 接口，
使其可以嵌入 OpticalLink 链路中。
"""
import numpy as np
from .base import OpticalComponent, OpticalSignal
from ..core.mrr_bank import MRRWeightBank


class MRRBankFilter(OpticalComponent):
    """
    MRR 权重阵列滤波器

    将 MRRWeightBank 的透射特性应用到光信号上。
    每个通道的功率被对应 MRR 的透过率加权。

    使用方式:
        >>> bank = MRRWeightBank(9, grid.wavelengths)
        >>> bank.program_weights(target_weights)
        >>> filt = MRRBankFilter(bank)
        >>> output = filt.forward(input_signal)
    """

    def __init__(self, weight_bank: MRRWeightBank):
        self.weight_bank = weight_bank

    @property
    def name(self) -> str:
        return f"MRRBank(N={self.weight_bank.num_weights})"

    def forward(self, signal: OpticalSignal) -> OpticalSignal:
        """
        对每个通道施加 MRR 透过率加权

        P_out[i] = P_in[i] × T_i(λ_i)
        """
        out = signal.copy()
        n = min(out.num_channels, self.weight_bank.num_weights)

        for i in range(n):
            t = self.weight_bank.mrrs[i].transmission(out.wavelengths[i])
            out.powers[i] *= t

        return out

    def get_weight_vector(self) -> np.ndarray:
        """获取当前权重向量 [-1, 1]"""
        return self.weight_bank.get_weights()
