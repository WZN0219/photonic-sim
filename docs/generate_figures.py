"""生成 README 所需的测试结果可视化图"""
import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from photonic_sim.core.physics import PhysicsConstants, voltage_to_shift
from photonic_sim.core.mrr import MRR
from photonic_sim.core.mrr_bank import MRRWeightBank
from photonic_sim.core.wavelength_grid import WavelengthGrid

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
OUT = "docs/figures"

import os
os.makedirs(OUT, exist_ok=True)

# ── 1. Lorentzian 透射谱 ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.5))
mrr = MRR(resonance_nm=1550.0)
wls = np.linspace(1549.0, 1551.5, 500)
# 不同加热器电压
for v, color in [(0, "#3b82f6"), (1.0, "#f59e0b"), (2.0, "#ef4444")]:
    mrr_copy = MRR(resonance_nm=1550.0)
    mrr_copy.set_heater(v)
    ts = [mrr_copy.transmission(w) for w in wls]
    ts_db = 10 * np.log10(np.array(ts) + 1e-10)
    ax.plot(wls, ts_db, color=color, linewidth=1.5,
            label=f"V={v:.1f}V (Δλ={mrr_copy.thermal_shift_nm:.3f}nm)")

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Transmission (dB)")
ax.set_title("MRR Lorentzian Transmission Spectrum")
ax.legend(fontsize=8)
ax.set_ylim(-30, 1)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/lorentzian_spectrum.png")
plt.close()
print("✓ lorentzian_spectrum.png")

# ── 2. V² 调谐曲线 ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))
pc = PhysicsConstants()
voltages = np.linspace(0, 5, 100)
shifts = [voltage_to_shift(v, pc.heater_resistance_ohm, pc.tuning_efficiency_nm_mw) for v in voltages]
ax.plot(voltages, shifts, color="#8b5cf6", linewidth=2)
ax.plot(voltages, [pc.tuning_efficiency_nm_mw * (v**2)/pc.heater_resistance_ohm*1000 for v in voltages],
        "--", color="#d946ef", linewidth=1, alpha=0.5, label="η·V²/R (analytical)")
ax.set_xlabel("Heater Voltage (V)")
ax.set_ylabel("Wavelength Shift (nm)")
ax.set_title("V² Thermal Tuning Law: Δλ = η·V²/R")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/v2_tuning.png")
plt.close()
print("✓ v2_tuning.png")

# ── 3. 串扰矩阵热图 ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4.5))
grid = WavelengthGrid()
bank = MRRWeightBank(num_weights=9, comb_wavelengths_nm=grid.wavelengths)
C = bank.crosstalk_matrix
im = ax.imshow(C, cmap="YlOrRd", vmin=0, vmax=0.1)
for i in range(9):
    for j in range(9):
        val = C[i, j]
        color = "white" if val > 0.05 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)
ax.set_title("N×N Thermal Crosstalk Matrix")
ax.set_xlabel("MRR index j (heater)")
ax.set_ylabel("MRR index i (affected)")
fig.colorbar(im, ax=ax, label="Crosstalk coefficient")
fig.tight_layout()
fig.savefig(f"{OUT}/crosstalk_matrix.png")
plt.close()
print("✓ crosstalk_matrix.png")

# ── 4. 权重编程对比 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

target = np.array([0.5, -0.3, 0.8, -0.1, 0.9, -0.7, 0.2, 0.4, -0.5])
bank_clean = MRRWeightBank(9, grid.wavelengths, drift_sigma_nm=0.0)
result = bank_clean.program_weights(target)
actual = np.array(result["actual_weights"])

x = np.arange(9)
axes[0].bar(x - 0.15, target, 0.3, color="#3b82f6", label="Target", alpha=0.8)
axes[0].bar(x + 0.15, actual, 0.3, color="#ef4444", label="Actual (with crosstalk)", alpha=0.8)
axes[0].set_xlabel("Weight Index")
axes[0].set_ylabel("Weight Value")
axes[0].set_title(f"Weight Programming (mean_err={result['mean_error']:.3f})")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# 误差分布
errors = np.abs(target - actual)
axes[1].bar(x, errors, color="#f59e0b", alpha=0.8)
axes[1].set_xlabel("Weight Index")
axes[1].set_ylabel("|Error|")
axes[1].set_title("Per-Weight Absolute Error")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"{OUT}/weight_programming.png")
plt.close()
print("✓ weight_programming.png")

# ── 5. 光链路功率传输 ────────────────────────────────────
from photonic_sim.components import (
    create_comb_signal, EOMModulator, MRRBankFilter,
    Photodetector, FiberSpan,
)
from photonic_sim.link import OpticalLink

signal = create_comb_signal(grid, power_per_line_mw=1.0, num_channels=9)
signal.data = np.array([0.8, 0.6, 0.9, 0.3, 1.0, 0.5, 0.7, 0.4, 0.2])

bank_link = MRRWeightBank(9, grid.wavelengths, drift_sigma_nm=0.0)
bank_link.program_weights(target)

stages = [
    ("Comb Source", signal.powers.copy()),
]

eom = EOMModulator(insertion_loss_db=2.0)
s1 = eom.load_data(signal, signal.data)
stages.append(("After EOM", s1.powers.copy()))

filt = MRRBankFilter(bank_link)
s2 = filt.forward(s1)
stages.append(("After MRR Bank", s2.powers.copy()))

pd = Photodetector(responsivity=0.8, adc_bits=10)
s3 = pd.forward(s2)
stages.append(("After PD", s3.powers.copy()))

fig, ax = plt.subplots(figsize=(8, 4))
for i, (name, vals) in enumerate(stages):
    ax.plot(range(9), vals, "o-", linewidth=1.5, markersize=5, label=name)
ax.set_xlabel("Channel Index")
ax.set_ylabel("Power (mW) / Current (mA)")
ax.set_title("Signal Power Through Optical Link")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/link_power.png")
plt.close()
print("✓ link_power.png")

print("\n所有图表已生成到 docs/figures/")
