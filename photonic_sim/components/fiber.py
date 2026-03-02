"""
光纤传输模型

模拟光纤中的信号衰减和受激拉曼散射 (SRS) 效应。
"""
import numpy as np
from .base import OpticalComponent, OpticalSignal


class FiberSpan(OpticalComponent):
    """
    光纤段

    物理模型:
      1. 线性衰减:  P_out = P_in × 10^(-α·L / 10)
      2. 连接器损耗: 每端 ~0.3 dB
      3. SRS 倾斜:  长波长功率增强，短波长功率减弱
         ΔP_i ∝ (λ_i - λ_center) × P_total × L × g_R

    Attributes:
        length_km:         光纤长度 [km]
        attenuation_db_km: 衰减系数 [dB/km] (典型 SMF: 0.2)
        connector_loss_db: 连接器损耗 [dB]
        srs_coefficient:   SRS 系数 (简化)
    """

    def __init__(self, length_km: float = 1.0,
                 attenuation_db_km: float = 0.2,
                 connector_loss_db: float = 0.3,
                 srs_coefficient: float = 0.001):
        self.length_km = length_km
        self.attenuation_db_km = attenuation_db_km
        self.connector_loss_db = connector_loss_db
        self.srs_coefficient = srs_coefficient

    @property
    def name(self) -> str:
        return f"Fiber({self.length_km}km, α={self.attenuation_db_km}dB/km)"

    def forward(self, signal: OpticalSignal) -> OpticalSignal:
        out = signal.copy()

        # 1. 线性衰减
        total_loss_db = (self.attenuation_db_km * self.length_km
                         + self.connector_loss_db)
        loss_linear = 10 ** (-total_loss_db / 10)
        out.powers = out.powers * loss_linear

        # 2. SRS 倾斜 (简化模型)
        if self.srs_coefficient > 0 and out.num_channels > 1:
            center_wl = np.mean(out.wavelengths)
            total_power = np.sum(out.powers)
            # SRS 使长波长增益、短波长损耗
            delta_wl = out.wavelengths - center_wl
            srs_factor = 1.0 + (self.srs_coefficient * delta_wl
                                * total_power * self.length_km)
            out.powers = out.powers * np.clip(srs_factor, 0.5, 2.0)

        return out
