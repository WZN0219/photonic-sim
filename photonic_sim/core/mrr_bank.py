"""
MRR 权重阵列 (Weight Bank)

核心改进 (相比 optical-agent 原始版本):
  1. 热串扰: N×N 指数衰减矩阵 (替代最近邻)
  2. 物理计算: 向量化矩阵运算 (替代逐环循环)
  3. 调谐规律: 通过 MRR 类的 V² 关系 (替代线性)

物理依据:
  - 串扰矩阵: C[i,j] = α × exp(-|i-j|/L), α~8%, L~2 index
  - Ref: Watts et al., Optics Express, 2013 (热串扰指数衰减)
  - Ref: Liu et al., Optica 2025 (ChiL 全局串扰建模)
"""
import numpy as np
from .mrr import MRR
from .physics import PhysicsConstants


class MRRWeightBank:
    """
    MRR 权重阵列 — 对应芯片上的 weight bank

    管理 N 个 MRR，模拟权重编程过程中的物理效应:
    串扰、随机漂移、全局温漂。

    Example:
        >>> from photonic_sim.core import WavelengthGrid, MRRWeightBank
        >>> grid = WavelengthGrid()
        >>> bank = MRRWeightBank(num_weights=9, comb_wavelengths_nm=grid.wavelengths)
        >>> result = bank.program_weights(np.zeros(9))
        >>> print(f"精度: {result['precision_bits']:.1f} bits")
    """

    def __init__(self, num_weights: int, comb_wavelengths_nm: np.ndarray,
                 crosstalk_alpha: float = None, crosstalk_decay: float = None,
                 drift_sigma_nm: float = None):
        """
        Args:
            num_weights:       权重数量 (= 卷积核大小, e.g. 9 for 3x3)
            comb_wavelengths_nm: 微梳各梳齿波长 [nm]
            crosstalk_alpha:   热串扰峰值 (默认 0.08)
            crosstalk_decay:   串扰衰减长度 (默认 2.0)
            drift_sigma_nm:    随机漂移标准差 [nm] (默认 0.005)
        """
        pc = PhysicsConstants()
        self.num_weights = num_weights
        self.comb_wavelengths = comb_wavelengths_nm[:num_weights].copy()

        # 每个权重对应一个 MRR
        self.mrrs = [MRR(resonance_nm=wl) for wl in self.comb_wavelengths]

        # 串扰参数
        self._ct_alpha = crosstalk_alpha if crosstalk_alpha is not None else pc.crosstalk_alpha
        self._ct_decay = crosstalk_decay if crosstalk_decay is not None else pc.crosstalk_decay_length

        # 构建 N×N 全局串扰矩阵
        self.crosstalk_matrix = self._build_crosstalk_matrix()

        # 噪声参数
        self.drift_sigma_nm = drift_sigma_nm if drift_sigma_nm is not None else pc.drift_sigma_nm
        self.global_temp_shift_nm: float = 0.0

        # 缓存
        self._cached_ideal_shifts = np.zeros(num_weights)
        self._cached_drift: np.ndarray = None  # 延迟初始化

    # ------------------------------------------------------------------ #
    #  串扰矩阵构建
    # ------------------------------------------------------------------ #

    def _build_crosstalk_matrix(self) -> np.ndarray:
        """
        构建 N×N 热串扰矩阵

        模型:
            C[i,i] = 1.0       (自身 100%)
            C[i,j] = α × exp(-|i-j| / L)   (i≠j)

        典型值 α=0.08, L=2.0:
            最近邻 (|i-j|=1): ~6.1%
            次近邻 (|i-j|=2): ~3.0%
            远距离 (|i-j|=4): ~1.1%
        """
        N = self.num_weights
        C = np.eye(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    C[i, j] = self._ct_alpha * np.exp(-abs(i - j) / self._ct_decay)
        return C

    # ------------------------------------------------------------------ #
    #  权重编程
    # ------------------------------------------------------------------ #

    def program_weights(self, target_weights: np.ndarray) -> dict:
        """
        编程目标权重

        流程:
          1. target_weight → target_T (线性映射)
          2. target_T → ideal_shift (Lorentzian 反解)
          3. ideal_shift → voltage → power (V² 关系)
          4. crosstalk_matrix × shifts → total_shifts (全局串扰)
          5. total_shifts + drift + global → effective_shifts
          6. 读回实际权重

        Args:
            target_weights: 目标权重值, shape=(num_weights,), 范围 [-1, 1]

        Returns:
            dict: 含 target/actual weights, error, precision_bits
        """
        target_weights = np.asarray(target_weights, dtype=float).flatten()
        assert len(target_weights) == self.num_weights

        # 1. 计算每个 MRR 的理想波长偏移
        ideal_shifts = np.array([
            mrr.get_shift_for_weight(w)
            for mrr, w in zip(self.mrrs, target_weights)
        ])
        self._cached_ideal_shifts = ideal_shifts.copy()

        # 2. 应用物理模型
        self._apply_physics_model(ideal_shifts)

        # 3. 读回实际权重
        actual_weights = self.get_weights()

        # 4. 计算误差
        clamped = np.clip(target_weights, -1.0, 1.0)
        error = np.abs(clamped - actual_weights)

        return {
            "target_weights": target_weights.tolist(),
            "actual_weights": [round(w, 4) for w in actual_weights],
            "max_error": round(float(np.max(error)), 4),
            "mean_error": round(float(np.mean(error)), 4),
            "precision_bits": round(float(-np.log2(np.mean(error) + 1e-10)), 1),
        }

    # ------------------------------------------------------------------ #
    #  物理引擎
    # ------------------------------------------------------------------ #

    def _apply_physics_model(self, ideal_shifts: np.ndarray):
        """
        向量化物理引擎

        核心公式:
            total_shift = C @ ideal_shifts + drift + global_temp_shift

        其中 C 是 N×N 串扰矩阵，@ 为矩阵乘法。
        """
        # 1. 全局热串扰: 矩阵乘法
        total_shifts = self.crosstalk_matrix @ ideal_shifts

        # 2. 随机漂移 (慢变化，缓存)
        if self._cached_drift is None:
            self._cached_drift = np.random.normal(
                0, self.drift_sigma_nm, self.num_weights
            )
        total_shifts += self._cached_drift

        # 3. 全局温漂
        total_shifts += self.global_temp_shift_nm

        # 4. 更新各 MRR 状态
        for i, mrr in enumerate(self.mrrs):
            # 反解电压（用于 get_status 显示）
            mrr.heater_voltage = mrr.get_voltage_for_shift(ideal_shifts[i])
            # 设置有效偏移（含串扰+漂移）
            mrr.thermal_shift_nm = total_shifts[i]

    # ------------------------------------------------------------------ #
    #  状态查询
    # ------------------------------------------------------------------ #

    def set_global_temp_shift(self, shift_nm: float):
        """设置全局温漂偏移 [nm] 并刷新物理状态"""
        self.global_temp_shift_nm = shift_nm
        self.refresh()

    def refresh(self):
        """刷新物理状态（温度变化后调用）"""
        self._apply_physics_model(self._cached_ideal_shifts)

    def reset_drift(self):
        """重新生成随机漂移（模拟时间流逝）"""
        self._cached_drift = np.random.normal(
            0, self.drift_sigma_nm, self.num_weights
        )
        self.refresh()

    def get_weights(self) -> np.ndarray:
        """读取当前实际权重值"""
        return np.array([
            mrr.weight_value(wl)
            for mrr, wl in zip(self.mrrs, self.comb_wavelengths)
        ])

    def get_status(self) -> dict:
        """获取阵列完整状态"""
        return {
            "num_weights": self.num_weights,
            "weights": [round(w, 4) for w in self.get_weights()],
            "heater_voltages": [round(m.heater_voltage, 3) for m in self.mrrs],
            "resonance_shifts_nm": [round(m.thermal_shift_nm, 4) for m in self.mrrs],
            "crosstalk_matrix_shape": list(self.crosstalk_matrix.shape),
        }
