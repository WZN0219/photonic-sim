import numpy as np


def build_comb_wavelengths(center_nm: float, fsr_nm: float, num_lines: int) -> np.ndarray:
    offsets = np.arange(num_lines) - num_lines // 2
    return center_nm + offsets * fsr_nm


def fold_detuning(delta_nm: np.ndarray, fsr_nm: float) -> np.ndarray:
    return (delta_nm + fsr_nm / 2.0) % fsr_nm - fsr_nm / 2.0


def lorentzian_transmission(delta_nm: np.ndarray, hwhm_nm: np.ndarray, min_t: float) -> np.ndarray:
    transmission = 1.0 - (1.0 - min_t) / (1.0 + (delta_nm / hwhm_nm) ** 2)
    return np.clip(transmission, min_t, 1.0)


def voltage_to_shift_nm(voltage_v: np.ndarray, resistance_ohm: float,
                        efficiency_nm_per_mw: float) -> np.ndarray:
    power_mw = (voltage_v ** 2) / resistance_ohm * 1000.0
    return efficiency_nm_per_mw * power_mw


def build_crosstalk_matrix(num_rings: int, alpha: float, decay_length: float) -> np.ndarray:
    idx = np.arange(num_rings)
    distance = np.abs(idx[:, None] - idx[None, :])
    matrix = alpha * np.exp(-distance / decay_length)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def adc_quantize_unipolar(values: np.ndarray, bits: int,
                          full_scale: float) -> tuple[np.ndarray, float]:
    if bits <= 0:
        return np.clip(values, 0.0, full_scale), full_scale
    levels = (2 ** bits) - 1
    lsb = full_scale / levels
    clipped = np.clip(values, 0.0, full_scale)
    quantized = np.round(clipped / lsb) * lsb
    return np.clip(quantized, 0.0, full_scale), lsb
