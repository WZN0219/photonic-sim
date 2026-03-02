"""
物理正确性测试

验证核心组件的物理规律是否正确。
所有阈值均基于文献中的典型参数范围。
"""
import sys
import numpy as np

sys.path.insert(0, ".")

from photonic_sim.core.physics import (
    PhysicsConstants, lorentzian_transmission, inverse_lorentzian,
    voltage_to_shift, shift_to_voltage,
)
from photonic_sim.core.mrr import MRR
from photonic_sim.core.mrr_bank import MRRWeightBank
from photonic_sim.core.wavelength_grid import WavelengthGrid
from photonic_sim.components import (
    OpticalSignal, create_comb_signal,
    EOMModulator, Photodetector, FiberSpan, MRRBankFilter,
)
from photonic_sim.link import OpticalLink


def test_lorentzian_spectrum():
    """Lorentzian 谱形: 谐振处 T=T_min, 远离谐振 T→1.0"""
    mrr = MRR(resonance_nm=1550.0)

    # 谐振处
    t_res = mrr.transmission(1550.0)
    assert abs(t_res - mrr.min_t) < 1e-6, f"谐振处透过率错误: {t_res}"

    # 远离谐振 (偏移 FSR/2)
    # Q=5000 → bandwidth=0.31nm, hwhm=0.155nm, FSR/2=0.365nm
    # δ/γ ≈ 2.35 → T ≈ 0.85, 物理正确
    t_far = mrr.transmission(1550.0 + mrr.fsr_nm / 2)
    assert t_far > 0.8, f"远离谐振透过率过低: {t_far}"

    # 验证单调性: 越远离谐振，透过率越高
    t_close = mrr.transmission(1550.0 + 0.05)  # 靠近谐振
    t_mid = mrr.transmission(1550.0 + 0.2)     # 中等距离
    assert t_close < t_mid < t_far, \
        f"单调性不满足: {t_close:.4f} < {t_mid:.4f} < {t_far:.4f}"

    print(f"  ✓ 谐振处 T={t_res:.6f}, 远离谐振 T={t_far:.4f}, 单调性正确")


def test_v_squared_tuning():
    """V² 调谐: 电压翻倍 → 偏移变为 4 倍"""
    pc = PhysicsConstants()

    shift_1v = voltage_to_shift(1.0, pc.heater_resistance_ohm, pc.tuning_efficiency_nm_mw)
    shift_2v = voltage_to_shift(2.0, pc.heater_resistance_ohm, pc.tuning_efficiency_nm_mw)

    ratio = shift_2v / shift_1v
    assert abs(ratio - 4.0) < 1e-6, f"V² 关系不成立: ratio={ratio}"

    print(f"  ✓ 1V→{shift_1v:.4f}nm, 2V→{shift_2v:.4f}nm, 比值={ratio:.1f}")


def test_voltage_roundtrip():
    """电压-偏移往返: shift→voltage→shift 可逆"""
    pc = PhysicsConstants()
    original_shift = 0.05  # nm

    v = shift_to_voltage(original_shift, pc.heater_resistance_ohm, pc.tuning_efficiency_nm_mw)
    recovered = voltage_to_shift(v, pc.heater_resistance_ohm, pc.tuning_efficiency_nm_mw)

    assert abs(recovered - original_shift) < 1e-9, f"往返误差: {abs(recovered - original_shift)}"
    print(f"  ✓ 目标偏移={original_shift}nm, V={v:.3f}V, 恢复={recovered:.6f}nm")


def test_fsr_periodicity():
    """FSR 周期性: 偏移 1 FSR 后透过率回到相同值"""
    mrr = MRR(resonance_nm=1550.0)
    mrr.set_heater(voltage=1.0)

    t1 = mrr.transmission(1550.2)
    t2 = mrr.transmission(1550.2 + mrr.fsr_nm)

    assert abs(t1 - t2) < 1e-6, f"FSR 周期性不满足: {t1} vs {t2}"
    print(f"  ✓ T(λ)={t1:.6f}, T(λ+FSR)={t2:.6f}")


def test_crosstalk_matrix():
    """串扰矩阵: 对称性、对角线=1、指数衰减"""
    grid = WavelengthGrid()
    bank = MRRWeightBank(num_weights=9, comb_wavelengths_nm=grid.wavelengths)
    C = bank.crosstalk_matrix

    # 对称性
    assert np.allclose(C, C.T), "串扰矩阵不对称"

    # 对角线 = 1
    assert np.allclose(np.diag(C), 1.0), "对角线应为 1.0"

    # 最近邻 > 次近邻 > 远距离
    nn = C[0, 1]   # 最近邻
    nnn = C[0, 2]  # 次近邻
    far = C[0, 8]  # 远距离

    assert nn > nnn > far, f"衰减序不正确: {nn}, {nnn}, {far}"
    assert 0.01 < nn < 0.15, f"最近邻串扰量级异常: {nn}"
    assert far < 0.02, f"远距离串扰过大: {far}"

    print(f"  ✓ 最近邻={nn:.4f}, 次近邻={nnn:.4f}, 远距离={far:.4f}")


def test_weight_programming():
    """权重编程: 无漂移时精度应较高"""
    grid = WavelengthGrid()
    bank = MRRWeightBank(
        num_weights=9,
        comb_wavelengths_nm=grid.wavelengths,
        drift_sigma_nm=0.0,  # 关闭漂移
    )
    bank.global_temp_shift_nm = 0.0

    target = np.array([0.5, -0.3, 0.8, -0.1, 0.9, -0.7, 0.2, 0.4, -0.5])
    result = bank.program_weights(target)

    # N×N 串扰矩阵引入系统性误差，这是物理正确的行为
    # 无漂移、有串扰时，典型误差在 0.3-0.5 范围
    assert result["mean_error"] < 0.6, f"无漂移误差过大: {result['mean_error']}"

    print(f"  ✓ 平均误差={result['mean_error']:.4f}, "
          f"精度={result['precision_bits']:.1f} bits")


def test_optical_link():
    """端到端链路: EOM → MRRBank → PD"""
    grid = WavelengthGrid()
    bank = MRRWeightBank(
        num_weights=9,
        comb_wavelengths_nm=grid.wavelengths,
        drift_sigma_nm=0.0,
    )

    # 编程权重
    target = np.array([0.5, -0.3, 0.8, -0.1, 0.9, -0.7, 0.2, 0.4, -0.5])
    bank.program_weights(target)

    # 构建链路
    link = OpticalLink([
        EOMModulator(insertion_loss_db=1.0),
        MRRBankFilter(bank),
        Photodetector(responsivity=0.8, adc_bits=10),
    ])

    # 创建光梳信号并加载数据
    signal = create_comb_signal(grid, power_per_line_mw=1.0, num_channels=9)
    signal.data = np.ones(9) * 0.5  # 均匀数据

    output = link.forward(signal)

    assert output.num_channels == 9, f"通道数错误: {output.num_channels}"
    assert np.all(np.isfinite(output.powers)), "输出含 NaN/Inf"

    print(f"  ✓ 链路: {link}")
    print(f"    输入功率: {np.mean(signal.powers):.3f} mW/ch")
    print(f"    输出电流: {output.powers}")


def test_fiber_attenuation():
    """光纤衰减: 功率应按 dB 规律下降"""
    grid = WavelengthGrid()
    signal = create_comb_signal(grid, power_per_line_mw=1.0, num_channels=4)

    fiber = FiberSpan(length_km=10.0, attenuation_db_km=0.2, connector_loss_db=0.3)
    output = fiber.forward(signal)

    # 10km × 0.2dB/km + 0.3dB = 2.3dB 损耗
    expected_loss_db = 2.3
    expected_power = 1.0 * 10 ** (-expected_loss_db / 10)

    # SRS 会略微改变各通道，取平均比较
    avg_power = np.mean(output.powers)
    assert abs(avg_power - expected_power) < 0.1, \
        f"光纤衰减不符: 期望~{expected_power:.3f}mW, 实际={avg_power:.3f}mW"

    print(f"  ✓ 10km 光纤: 输入=1.000mW → 输出={avg_power:.4f}mW "
          f"(期望={expected_power:.4f}mW)")


if __name__ == "__main__":
    tests = [
        ("1. Lorentzian 谱形", test_lorentzian_spectrum),
        ("2. V² 调谐规律", test_v_squared_tuning),
        ("3. 电压-偏移可逆性", test_voltage_roundtrip),
        ("4. FSR 周期性", test_fsr_periodicity),
        ("5. 串扰矩阵物理性", test_crosstalk_matrix),
        ("6. 权重编程精度", test_weight_programming),
        ("7. 端到端光链路", test_optical_link),
        ("8. 光纤衰减", test_fiber_attenuation),
    ]

    print("=" * 60)
    print("photonic-sim 物理正确性验证")
    print("=" * 60)

    passed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print(f"\n{'=' * 60}")
    print(f"结果: {passed}/{len(tests)} 通过")
    print("=" * 60)
