"""
光信号创建工具

提供从 WavelengthGrid 生成标准光信号的辅助函数。
"""
import numpy as np
from .base import OpticalSignal
from ..core.wavelength_grid import WavelengthGrid


def create_comb_signal(grid: WavelengthGrid, power_per_line_mw: float = 1.0,
                       num_channels: int = None) -> OpticalSignal:
    """
    从微梳波长网格创建均匀功率的光梳信号

    Args:
        grid:               波长网格
        power_per_line_mw:  每根梳齿功率 [mW]
        num_channels:       使用的通道数 (默认全部)

    Returns:
        均匀功率的 OpticalSignal
    """
    n = num_channels or grid.num_lines
    wavelengths = grid.get_channel_wavelengths(n)
    powers = np.full(n, power_per_line_mw)
    return OpticalSignal(wavelengths=wavelengths, powers=powers)
