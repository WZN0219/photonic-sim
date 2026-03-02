"""
微梳波长网格管理

基于 Bai et al., Nature Communications 2023 的 AlGaAsOI 微梳参数。
微梳产生等间距梳齿，每根梳齿对应一个 WDM 权重通道。
"""
import numpy as np
from .physics import PhysicsConstants


class WavelengthGrid:
    """
    微梳波长网格

    生成以 center_nm 为中心、间距为 fsr_nm 的等间距梳齿波长。

    Attributes:
        center_nm:   中心波长 [nm]
        fsr_nm:      梳齿间距 [nm]
        num_lines:   梳齿总数
        wavelengths: 梳齿波长数组 [nm], shape=(num_lines,)
    """

    def __init__(self, center_nm: float = None, fsr_nm: float = None,
                 num_lines: int = 20):
        pc = PhysicsConstants()
        self.center_nm = center_nm or pc.default_resonance_nm
        self.fsr_nm = fsr_nm or pc.default_fsr_nm
        self.num_lines = num_lines

        offsets = np.arange(num_lines) - num_lines // 2
        self.wavelengths = self.center_nm + offsets * self.fsr_nm

    def get_channel_wavelengths(self, num_channels: int) -> np.ndarray:
        """获取前 num_channels 根梳齿波长"""
        if num_channels > self.num_lines:
            raise ValueError(
                f"需要 {num_channels} 个通道，但微梳只有 {self.num_lines} 根梳齿"
            )
        return self.wavelengths[:num_channels].copy()

    def __repr__(self) -> str:
        return (f"WavelengthGrid(center={self.center_nm:.1f}nm, "
                f"FSR={self.fsr_nm:.2f}nm, lines={self.num_lines})")
