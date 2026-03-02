"""
单个微环谐振器 (MRR) 模型

物理模型:
  1. Lorentzian 透射谱:  T(λ) = 1 - (1-T_min)/(1+(δ/γ)²)
  2. V² 热光调谐:        Δλ = η × V²/R
  3. FSR 周期折叠:        δ 对 FSR 取模

参数来源:
  - Bogaerts et al., Laser & Photonics Reviews, 2012
  - Bai et al., Nature Communications, 2023
  - Huang et al., APL Photonics, 2020
"""
import numpy as np
from .physics import (
    PhysicsConstants,
    lorentzian_transmission,
    inverse_lorentzian,
    voltage_to_shift,
    shift_to_voltage,
)


class MRR:
    """
    单个微环谐振器权重单元

    模拟 MRR 的透射特性和热光调谐行为。
    加热器通过 V² 功率关系改变谐振波长位置，从而调节透过率/权重。

    Example:
        >>> mrr = MRR(resonance_nm=1550.0)
        >>> mrr.set_heater(voltage=2.0)
        >>> print(f"透过率: {mrr.transmission(1550.0):.4f}")
    """

    def __init__(self, resonance_nm: float = None, fsr_nm: float = None,
                 extinction_ratio_db: float = None, q_factor: float = None,
                 heater_resistance_ohm: float = None,
                 tuning_efficiency_nm_mw: float = None):
        pc = PhysicsConstants()

        self.resonance_nm = resonance_nm or pc.default_resonance_nm
        self.fsr_nm = fsr_nm or pc.default_fsr_nm
        self.er_db = extinction_ratio_db or pc.default_er_db
        self.q_factor = q_factor or pc.default_q_factor
        self.heater_resistance_ohm = heater_resistance_ohm or pc.heater_resistance_ohm
        self.tuning_efficiency_nm_mw = tuning_efficiency_nm_mw or pc.tuning_efficiency_nm_mw

        # 导出参数
        self.bandwidth_nm = self.resonance_nm / self.q_factor  # 3dB 带宽
        self.min_t = 10 ** (-self.er_db / 10)                  # 谐振处最小透过率

        # 状态
        self.heater_voltage: float = 0.0
        self.thermal_shift_nm: float = 0.0  # 最终有效偏移（含串扰、漂移）

    @property
    def hwhm_nm(self) -> float:
        """半高半宽 [nm]"""
        return self.bandwidth_nm / 2.0

    @property
    def effective_resonance(self) -> float:
        """当前有效谐振波长 [nm] (含热偏移)"""
        return self.resonance_nm + self.thermal_shift_nm

    def set_heater(self, voltage: float):
        """
        设置加热器电压 (V² 调谐)

        物理过程: V → P=V²/R → Δλ=η×P
        """
        self.heater_voltage = voltage
        self.thermal_shift_nm = voltage_to_shift(
            voltage, self.heater_resistance_ohm, self.tuning_efficiency_nm_mw
        )

    def transmission(self, wavelength_nm: float) -> float:
        """
        计算给定波长处的透过率

        步骤:
          1. 计算波长失谐 δ = λ - λ_res_eff
          2. FSR 周期折叠: δ mod FSR
          3. Lorentzian 公式计算透过率
        """
        delta = wavelength_nm - self.effective_resonance
        # FSR 周期折叠到 [-FSR/2, FSR/2]
        delta = (delta + self.fsr_nm / 2) % self.fsr_nm - self.fsr_nm / 2
        return lorentzian_transmission(delta, self.hwhm_nm, self.min_t)

    def weight_value(self, wavelength_nm: float) -> float:
        """
        将透过率线性映射为权重值 [-1, 1]

        映射: T_min → -1,  1.0 → +1
        """
        t = self.transmission(wavelength_nm)
        return 2.0 * (t - self.min_t) / (1.0 - self.min_t) - 1.0

    def get_shift_for_weight(self, target_weight: float) -> float:
        """
        计算达到目标权重所需的波长偏移量

        步骤:
          1. weight → target_T (反映射)
          2. target_T → δ (Lorentzian 反函数)

        Returns:
            所需的波长偏移 [nm]
        """
        w_clamped = np.clip(target_weight, -1.0, 1.0)
        t_target = self.min_t + (w_clamped + 1.0) / 2.0 * (1.0 - self.min_t)
        return inverse_lorentzian(t_target, self.hwhm_nm, self.min_t)

    def get_voltage_for_shift(self, shift_nm: float) -> float:
        """计算产生目标偏移所需的电压 [V]"""
        return shift_to_voltage(
            shift_nm, self.heater_resistance_ohm, self.tuning_efficiency_nm_mw
        )

    def __repr__(self) -> str:
        return (f"MRR(λ_res={self.resonance_nm:.2f}nm, "
                f"Q={self.q_factor:.0f}, ER={self.er_db:.0f}dB)")
