"""
物理常量与工具函数

硅光子平台微环谐振器相关的物理参数，参考值来源：
- 热光系数: Komma et al., APL 2012
- 典型 MRR 参数: Bogaerts et al., Laser & Photonics Reviews 2012
- 微梳参数: Bai et al., Nature Communications 2023
"""
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsConstants:
    """硅光子 MRR 物理常量"""

    # --- 材料参数 (SOI @1550nm) ---
    thermo_optic_coeff: float = 1.86e-4      # dn/dT [K⁻¹]
    n_eff: float = 2.4                        # 有效折射率
    n_group: float = 4.2                      # 群折射率

    # --- MRR 默认参数 ---
    default_resonance_nm: float = 1550.0
    default_fsr_nm: float = 0.73              # ~91 GHz @1550nm
    default_fsr_ghz: float = 91.0
    default_q_factor: float = 5000.0
    default_er_db: float = 25.0

    # --- 加热器参数 ---
    heater_resistance_ohm: float = 1200.0     # [Ω]
    tuning_efficiency_nm_mw: float = 0.015    # [nm/mW]

    # --- 热串扰参数 ---
    crosstalk_alpha: float = 0.08             # 最近邻串扰峰值
    crosstalk_decay_length: float = 2.0       # 衰减长度 [index]

    # --- 噪声参数 ---
    drift_sigma_nm: float = 0.005             # 随机漂移 [nm]
    thermal_drift_rate_nm_per_k: float = 0.01 # 温漂系数 [nm/K]


def lorentzian_transmission(delta_nm: float, hwhm_nm: float, min_t: float) -> float:
    """
    Lorentzian 透射函数 (notch filter)

        T(δ) = 1 - (1 - T_min) / (1 + (δ/γ)²)

    Args:
        delta_nm: 波长失谐 [nm]
        hwhm_nm: 半高半宽 [nm]
        min_t:   谐振处最小透过率

    Ref: Bogaerts et al., Laser & Photonics Reviews, 2012
    """
    t = 1.0 - (1.0 - min_t) / (1.0 + (delta_nm / hwhm_nm) ** 2)
    return float(np.clip(t, min_t, 1.0))


def inverse_lorentzian(t_target: float, hwhm_nm: float, min_t: float) -> float:
    """
    Lorentzian 反函数：目标透过率 → 波长失谐

        δ = γ × sqrt((1 - T_min)/(1 - T) - 1)
    """
    if t_target >= 1.0 - 1e-9:
        return hwhm_nm * 10.0
    if t_target <= min_t + 1e-9:
        return 0.0
    ratio = (1.0 - min_t) / (1.0 - t_target) - 1.0
    return hwhm_nm * np.sqrt(max(ratio, 0.0))


def voltage_to_shift(voltage: float, resistance_ohm: float,
                     efficiency_nm_mw: float) -> float:
    """
    V² 调谐：电压 → 波长红移

        P = V²/R [W],  Δλ = η × P × 1000 [nm]

    Ref: Huang et al., APL Photonics 2020, DOI:10.1063/1.5144121
    """
    power_mw = (voltage ** 2) / resistance_ohm * 1000.0
    return efficiency_nm_mw * power_mw


def shift_to_voltage(shift_nm: float, resistance_ohm: float,
                     efficiency_nm_mw: float) -> float:
    """反函数：目标波长偏移 → 所需电压  V = sqrt(Δλ × R / (η × 1000))"""
    if shift_nm <= 0:
        return 0.0
    power_mw = shift_nm / efficiency_nm_mw
    return np.sqrt(power_mw * resistance_ohm / 1000.0)
