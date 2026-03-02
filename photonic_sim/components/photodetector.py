"""
光电探测器 (Photodetector)

将多波长光信号转换为电信号，包含散粒噪声和 ADC 量化。
"""
import numpy as np
from .base import OpticalComponent, OpticalSignal


class Photodetector(OpticalComponent):
    """
    光电探测器

    物理模型:
      1. 光电转换:  I = R × P           (R: 响应度 A/W)
      2. 散粒噪声:  σ = sqrt(2qI·BW)    (简化为比例噪声)
      3. ADC 量化:  离散化到 2^bits 级

    Attributes:
        responsivity:   响应度 [A/W]
        dark_current_na: 暗电流 [nA]
        noise_sigma:    噪声相对强度
        adc_bits:       ADC 位数
    """

    def __init__(self, responsivity: float = 0.8,
                 dark_current_na: float = 1.0,
                 noise_sigma: float = 0.002,
                 adc_bits: int = 10):
        self.responsivity = responsivity
        self.dark_current_na = dark_current_na
        self.noise_sigma = noise_sigma
        self.adc_bits = adc_bits

    @property
    def name(self) -> str:
        return f"PD(R={self.responsivity}A/W, {self.adc_bits}bit)"

    def forward(self, signal: OpticalSignal) -> OpticalSignal:
        """
        光电转换

        Returns:
            输出信号的 powers 字段替换为电流值 [mA]
        """
        out = signal.copy()

        # 1. 光电转换: I = R × P [mA] (P in mW)
        photocurrent = self.responsivity * out.powers

        # 2. 散粒噪声 (简化为高斯)
        noise = np.random.normal(
            0, self.noise_sigma * np.abs(photocurrent) + 1e-6,
            size=photocurrent.shape
        )
        photocurrent += noise

        # 3. 暗电流
        photocurrent += self.dark_current_na * 1e-6  # nA → mA

        # 4. ADC 量化
        if self.adc_bits > 0:
            max_val = np.max(np.abs(photocurrent)) + 1e-9
            levels = 2 ** self.adc_bits
            step = 2 * max_val / levels
            photocurrent = np.round(photocurrent / step) * step

        out.powers = photocurrent
        return out

    def detect_sum(self, signal: OpticalSignal) -> float:
        """
        便捷方法：光电转换后对所有通道电流求和

        用于模拟 PPU 中 PD 将多波长加权信号汇聚为单一电信号。
        """
        result = self.forward(signal)
        return float(np.sum(result.powers))
