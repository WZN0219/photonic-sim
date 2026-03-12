import argparse
import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PALETTE = ["#005f73", "#0a9396", "#ee9b00", "#bb3e03", "#9b2226", "#3a86ff", "#8338ec"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an HTML report for experiment CSV outputs.")
    parser.add_argument("--outputs-dir", default="experiments/outputs")
    parser.add_argument("--output", default="experiments/outputs/experiment_report.html")
    return parser.parse_args()


def load_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: convert(value) for key, value in row.items()} for row in reader]


def convert(value):
    text = "" if value is None else value.strip()
    if text == "True":
        return True
    if text == "False":
        return False
    try:
        return float(text) if any(token in text for token in (".", "e", "E")) else int(text)
    except ValueError:
        return text


def esc(value):
    return html.escape(str(value))


def fmt(value, digits=3):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def mix(low: str, high: str, t: float):
    t = max(0.0, min(1.0, t))
    lo = [int(low[i : i + 2], 16) for i in (1, 3, 5)]
    hi = [int(high[i : i + 2], 16) for i in (1, 3, 5)]
    rgb = [round(lo[i] + (hi[i] - lo[i]) * t) for i in range(3)]
    return "#" + "".join(f"{value:02x}" for value in rgb)


def nice_step(raw):
    if raw <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw))
    fraction = raw / (10 ** exponent)
    if fraction <= 1:
        nice = 1
    elif fraction <= 2:
        nice = 2
    elif fraction <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exponent)


def ticks(vmin, vmax, count=5):
    if vmin == vmax:
        return [vmin]
    step = nice_step((vmax - vmin) / max(count - 1, 1))
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    values = []
    current = start
    while current <= end + step * 0.5 and len(values) < 100:
        values.append(round(current, 10))
        current += step
    return values


def line_chart(series, x_label, y_label, width=760, height=320):
    points = [pt for item in series for pt in item["points"]]
    if not points:
        return "<p>No data.</p>"
    ml, mr, mt, mb = 72, 16, 20, 48
    pw, ph = width - ml - mr, height - mt - mb
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmin -= 1
        xmax += 1
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    xt = ticks(xmin, xmax, 6)
    yt = ticks(ymin, ymax, 6)

    def sx(value):
        return ml + (value - xmin) / (xmax - xmin) * pw

    def sy(value):
        return mt + ph - (value - ymin) / (ymax - ymin) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart-svg">']
    for tick in yt:
        y = sy(tick)
        parts.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{width - mr}" y2="{y:.2f}" class="grid" />')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end" class="axis">{esc(fmt(tick))}</text>')
    for tick in xt:
        x = sx(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{height - mb}" class="grid" />')
        parts.append(f'<text x="{x:.2f}" y="{height - mb + 18}" text-anchor="middle" class="axis">{esc(fmt(tick))}</text>')
    parts.append(f'<line x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}" class="axis-line" />')
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height - mb}" class="axis-line" />')
    for item in series:
        polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in item["points"])
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{item["color"]}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" />')
    legend_x = ml
    for item in series:
        parts.append(f'<rect x="{legend_x}" y="8" width="14" height="14" rx="3" fill="{item["color"]}" />')
        parts.append(f'<text x="{legend_x + 20}" y="19" class="legend">{esc(item["name"])}</text>')
        legend_x += 24 + len(item["name"]) * 7
    parts.append(f'<text x="{ml + pw / 2:.2f}" y="{height - 10}" text-anchor="middle" class="label">{esc(x_label)}</text>')
    parts.append(f'<text x="18" y="{mt + ph / 2:.2f}" text-anchor="middle" transform="rotate(-90 18 {mt + ph / 2:.2f})" class="label">{esc(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(labels, values, y_label, width=760, height=320, colors=None):
    if not labels:
        return "<p>No data.</p>"
    colors = colors or [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    ml, mr, mt, mb = 72, 16, 20, 60
    pw, ph = width - ml - mr, height - mt - mb
    ymax = max(max(values), 1.0)
    yt = ticks(0.0, ymax, 6)
    top = max(yt)

    def sy(value):
        return mt + ph - (value / top) * ph

    slot = pw / len(labels)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart-svg">']
    for tick in yt:
        y = sy(tick)
        parts.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{width - mr}" y2="{y:.2f}" class="grid" />')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.2f}" text-anchor="end" class="axis">{esc(fmt(tick))}</text>')
    parts.append(f'<line x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}" class="axis-line" />')
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height - mb}" class="axis-line" />')
    for index, (label, value) in enumerate(zip(labels, values)):
        x = ml + index * slot + slot * 0.18
        w = slot * 0.64
        y = sy(value)
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{height - mb - y:.2f}" rx="4" fill="{colors[index]}" />')
        parts.append(f'<text x="{x + w / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" class="axis">{esc(fmt(value))}</text>')
        parts.append(f'<text x="{x + w / 2:.2f}" y="{height - mb + 20}" text-anchor="middle" class="axis">{esc(label)}</text>')
    parts.append(f'<text x="18" y="{mt + ph / 2:.2f}" text-anchor="middle" transform="rotate(-90 18 {mt + ph / 2:.2f})" class="label">{esc(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def heatmap(x_labels, y_labels, values, value_label, low="#f8fafc", high="#0f766e", notes=None, width=760, height=320, x_every=1):
    if not x_labels or not y_labels:
        return "<p>No data.</p>"
    ml, mr, mt, mb = 88, 16, 20, 60
    pw, ph = width - ml - mr, height - mt - mb
    cols, rows = len(x_labels), len(y_labels)
    cw, ch = pw / cols, ph / rows
    flat = [cell for row in values for cell in row]
    vmin, vmax = min(flat), max(flat)
    if vmin == vmax:
        vmax = vmin + 1.0
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart-svg">']
    for r, row_label in enumerate(y_labels):
        y = mt + r * ch
        parts.append(f'<text x="{ml - 10}" y="{y + ch / 2 + 4:.2f}" text-anchor="end" class="axis">{esc(row_label)}</text>')
        for c, value in enumerate(values[r]):
            x = ml + c * cw
            color = mix(low, high, (value - vmin) / (vmax - vmin))
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw:.2f}" height="{ch:.2f}" fill="{color}" stroke="#ffffff" stroke-width="1" />')
            if notes:
                label = notes[r][c]
                if label:
                    parts.append(f'<text x="{x + cw / 2:.2f}" y="{y + ch / 2 + 4:.2f}" text-anchor="middle" class="heat">{esc(label)}</text>')
    for c, label in enumerate(x_labels):
        if c % x_every == 0:
            parts.append(f'<text x="{ml + c * cw + cw / 2:.2f}" y="{height - mb + 18}" text-anchor="middle" class="axis">{esc(label)}</text>')
    parts.append(f'<text x="{ml}" y="{height - 12}" class="legend">{esc(value_label)} min={esc(fmt(vmin))} max={esc(fmt(vmax))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def card(label, value, help_text):
    return f'<div class="card"><div class="card-k">{esc(label)}</div><div class="card-v">{esc(value)}</div><div class="card-h">{esc(help_text)}</div></div>'


def section(title, subtitle, copy, cards_html, charts_html):
    section_id = title.lower().replace(" ", "-")
    return (
        f'<section id="{section_id}" class="section">'
        f'<h2>{esc(title)}</h2>'
        f'<p class="subtitle">{esc(subtitle)}</p>'
        f'<p class="copy">{copy}</p>'
        f'<div class="cards">{cards_html}</div>'
        f'<div class="charts">{charts_html}</div>'
        "</section>"
    )


def build_step(outputs_dir: Path):
    rows = load_csv(outputs_dir / "step_response.csv")
    final_shift = rows[-1]["shift_nm"]
    rise_90 = next(row["time_ms"] for row in rows if row["shift_nm"] >= final_shift * 0.9)
    cards_html = "".join(
        [
            card("目标通道", rows[0]["channel"], "单个 ring 被施加阶跃输入。"),
            card("最终偏移", f"{final_shift * 1e3:.2f} pm", "看热稳态最终到哪里。"),
            card("90% 上升时间", f"{fmt(rise_90, 1)} ms", "看响应速度是否够快。"),
        ]
    )
    charts = (
        '<div class="chart">'
        + line_chart(
            [{"name": "Voltage (V)", "color": PALETTE[0], "points": [(row["time_ms"], row["actuator_voltage_v"]) for row in rows]}],
            "Time (ms)",
            "Actuator Voltage (V)",
        )
        + "</div>"
        + '<div class="chart">'
        + line_chart(
            [{"name": "Thermal Power (mW)", "color": PALETTE[2], "points": [(row["time_ms"], row["thermal_power_mw"]) for row in rows]}],
            "Time (ms)",
            "Thermal Power (mW)",
        )
        + "</div>"
        + '<div class="chart">'
        + line_chart(
            [{"name": "Shift (pm)", "color": PALETTE[4], "points": [(row["time_ms"], row["shift_nm"] * 1e3) for row in rows]}],
            "Time (ms)",
            "Resonance Shift (pm)",
        )
        + "</div>"
    )
    return section(
        "Step Response",
        "执行器动态辨识",
        "对应单通道执行器阶跃响应测试。核心是在固定目标电压下，看执行器电压、热功率和谐振偏移如何随时间收敛，从而判断一阶动力学是否足够描述执行器。",
        cards_html,
        charts,
    )


def build_crosstalk(outputs_dir: Path):
    rows = load_csv(outputs_dir / "crosstalk_scan.csv")
    ring_cols = sorted([key for key in rows[0] if key.startswith("shift_ring_")], key=lambda item: int(item.split("_")[2]))
    drive_channel = rows[0]["drive_channel"]
    matrix = [[row[col] * 1e3 for col in ring_cols] for row in rows]
    max_shift = [rows[-1][col] * 1e3 for col in ring_cols]
    self_shift = max_shift[drive_channel]
    neighbor_ratio = max_shift[drive_channel - 1] / self_shift if drive_channel > 0 else 0.0
    cards_html = "".join(
        [
            card("驱动 ring", f"R{drive_channel}", "扫描时只驱动这一通道。"),
            card("主通道偏移", f"{self_shift:.2f} pm", "最大驱动下主通道的热漂移。"),
            card("最近邻串扰比", f"{neighbor_ratio * 100:.1f}%", "邻近 ring 相对主通道的偏移比例。"),
        ]
    )
    heat = heatmap(
        [f"R{i}" for i in range(len(ring_cols))],
        [fmt(row["drive_voltage_v"], 1) for row in rows],
        matrix,
        "Shift (pm)",
        low="#f8fafc",
        high="#c2410c",
    )
    bars = bar_chart(
        [f"R{i}" for i in range(len(ring_cols))],
        [round(value, 3) for value in max_shift],
        "Max-drive shift (pm)",
        colors=[PALETTE[4] if i == drive_channel else PALETTE[1] for i in range(len(ring_cols))],
    )
    return section(
        "Crosstalk Scan",
        "热串扰定量扫描",
        "对应单通道驱动、全阵列观测测试。它回答的是：给某个 ring 加热以后，其余 ring 会被带偏多少，以及这种影响是否随着距离快速衰减。",
        cards_html,
        f'<div class="chart">{heat}</div><div class="chart">{bars}</div>',
    )


def build_observation(outputs_dir: Path):
    pd_rows = load_csv(outputs_dir / "observation_chain" / "pd_sweep.csv")
    osa_rows = load_csv(outputs_dir / "observation_chain" / "osa_sweep.csv")
    adc_bits = sorted({row["adc_bits"] for row in pd_rows})
    full_scales = sorted({row["full_scale_current_ma"] for row in pd_rows})
    input_powers = sorted({row["input_power_mw"] for row in pd_rows})
    pd_lookup = {(row["adc_bits"], row["full_scale_current_ma"], row["input_power_mw"]): row for row in pd_rows}
    pd_panels = []
    sat_rate_1mw = 0.0
    for power in input_powers:
        values = []
        notes = []
        scoped = [row for row in pd_rows if row["input_power_mw"] == power]
        if power == 1.0:
            sat_rate_1mw = sum(bool(row["saturated"]) for row in scoped) / len(scoped)
        for bits in adc_bits:
            row_values = []
            row_notes = []
            for scale in full_scales:
                record = pd_lookup[(bits, scale, power)]
                usage = min(record["max_quantized_current_ma"] / scale, 1.0)
                row_values.append(usage)
                row_notes.append("SAT" if record["saturated"] else f"{usage * 100:.0f}%")
            values.append(row_values)
            notes.append(row_notes)
        pd_panels.append(
            '<div class="chart">'
            f"<h3>PD Dynamic Range @ {fmt(power, 1)} mW</h3>"
            + heatmap([fmt(scale, 3) for scale in full_scales], [str(bits) for bits in adc_bits], values, "Max current / full-scale", low="#ecfdf5", high="#b91c1c", notes=notes)
            + "</div>"
        )
    steps = sorted({row["step_pm"] for row in osa_rows})
    spans = sorted({row["span_nm"] for row in osa_rows})
    sample_lookup = {(row["step_pm"], row["span_nm"]): row["num_samples"] for row in osa_rows}
    sample_values = []
    sample_notes = []
    for step in steps:
        v_row = []
        n_row = []
        for span in spans:
            samples = sample_lookup[(step, span)]
            v_row.append(samples)
            n_row.append(str(samples))
        sample_values.append(v_row)
        sample_notes.append(n_row)
    frame_periods = sorted({row["frame_period_ms"] for row in osa_rows})
    stale_rates = []
    for period in frame_periods:
        scoped = [row for row in osa_rows if row["frame_period_ms"] == period]
        stale_rates.append(100.0 * sum(row["second_frame_quality"] != "fresh" for row in scoped) / len(scoped))
    cards_html = "".join(
        [
            card("PD 饱和率 @ 1 mW", f"{sat_rate_1mw * 100:.1f}%", "同一输入功率下，多少配置会顶满量程。"),
            card("OSA 最大采样点数", max(sample_lookup.values()), "由 step_pm 和 span_nm 共同决定。"),
            card("第二帧 stale 率", f"{max(stale_rates):.0f}%", "这里第二次采样只隔 1 ms，所以默认配置全 stale。"),
        ]
    )
    charts = "".join(pd_panels) + '<div class="chart">' + heatmap([fmt(span, 1) for span in spans], [fmt(step, 1) for step in steps], sample_values, "OSA sample count", low="#eff6ff", high="#1d4ed8", notes=sample_notes) + '</div>' + '<div class="chart">' + bar_chart([f"{fmt(period, 1)} ms" for period in frame_periods], [round(rate, 1) for rate in stale_rates], "Second-frame stale rate (%)", colors=[PALETTE[5] for _ in stale_rates]) + "</div>"
    return section(
        "Observation Chain Sweep",
        "PD / OSA 参数可观测性测试",
        "对应观测链扫频测试。前半段看 PD 的位宽与满量程是否会因为量化或饱和而误导判断，后半段看 OSA 的采样步进、span 和帧周期怎样影响可观测性与刷新时序。",
        cards_html,
        charts,
    )


def build_drift(outputs_dir: Path):
    dataset_dir = outputs_dir / "drift_dataset"
    latent_rows = load_csv(dataset_dir / "latent_state.csv")
    pd_rows = load_csv(dataset_dir / "pd_frames.csv")
    osa_rows = load_csv(dataset_dir / "osa_frames.csv")
    shift_cols = sorted([key for key in latent_rows[0] if key.startswith("shift_nm_")], key=lambda item: int(item.split("_")[-1]))
    pd_cols = sorted([key for key in pd_rows[0] if key.startswith("pd_q_ma_")], key=lambda item: int(item.split("_")[-1]))
    latent_series = [{"name": f"Ring {i}", "color": PALETTE[i % len(PALETTE)], "points": [(row["time_ms"], row[col] * 1e3) for row in latent_rows]} for i, col in enumerate(shift_cols)]
    pd_series = [{"name": f"PD {i}", "color": PALETTE[i % len(PALETTE)], "points": [(row["timestamp_ms"], row[col]) for row in pd_rows]} for i, col in enumerate(pd_cols)]
    timestamps = sorted({row["timestamp_ms"] for row in osa_rows})
    wavelengths = sorted({row["wavelength_nm"] for row in osa_rows})
    osa_lookup = {(row["timestamp_ms"], row["wavelength_nm"]): row["spectrum_dbm"] for row in osa_rows}
    spectrum = [[osa_lookup[(timestamp, wavelength)] for wavelength in wavelengths] for timestamp in timestamps]
    drift_span = max(max(row[col] for row in latent_rows) - min(row[col] for row in latent_rows) for col in shift_cols) * 1e3
    cards_html = "".join(
        [
            card("数据时长", f"{fmt(latent_rows[-1]['time_ms'], 1)} ms", "latent 状态连续记录的总长度。"),
            card("最大漂移范围", f"{drift_span:.2f} pm", "所有 ring 中最大 shift 变化范围。"),
            card("观测帧数", f"PD {len(pd_rows)} / OSA {len(timestamps)}", "多速率观测流的样本数。"),
        ]
    )
    charts = (
        '<div class="chart">' + line_chart(latent_series, "Time (ms)", "Latent shift (pm)") + "</div>"
        + '<div class="chart">' + line_chart(pd_series, "Timestamp (ms)", "PD quantized current (mA)") + "</div>"
        + '<div class="chart">' + heatmap([fmt(w, 3) for w in wavelengths], [fmt(t, 1) for t in timestamps], spectrum, "OSA spectrum (dBm)", low="#0f172a", high="#f59e0b", x_every=max(1, len(wavelengths) // 8)) + "</div>"
    )
    return section(
        "Drift Observation Dataset",
        "长时漂移与多观测流对齐",
        "对应长时间漂移数据集测试。它让系统在随机热漂移下持续演化，同时导出 latent、PD、OSA 三条观测流，供后续 bootstrap、滤波或状态估计算法使用。",
        cards_html,
        charts,
    )


def build_html(outputs_dir: Path):
    overview_rows = [
        ("`run_step_response.py`", "单通道执行器阶跃响应", "验证执行器动态是否可辨识。"),
        ("`run_crosstalk_scan.py`", "全阵列热串扰扫描", "量化单点驱动对其它 ring 的影响。"),
        ("`run_observation_chain_sweep.py`", "PD / OSA 参数扫频", "判断观测链参数会不会把结论带偏。"),
        ("`run_drift_observation_dataset.py`", "长时漂移数据集生成", "为后续 CalibrationBootstrap 准备多源观测样本。"),
    ]
    table_rows = "".join(f"<tr><td>{esc(name)}</td><td>{esc(test)}</td><td>{esc(goal)}</td></tr>" for name, test, goal in overview_rows)
    sections = "".join(
        [
            build_step(outputs_dir),
            build_crosstalk(outputs_dir),
            build_observation(outputs_dir),
            build_drift(outputs_dir),
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Photonic Sim Experiment Report</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --paper: #fffaf1;
      --card: #fffdf7;
      --line: #d4c7b5;
      --ink: #1f2937;
      --muted: #6b7280;
      --accent: #8f3b1b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #f8e8cf 0, transparent 30%),
        linear-gradient(180deg, #efe3d0 0%, var(--bg) 28%, #efe7db 100%);
    }}
    .page {{ width: min(1220px, calc(100vw - 28px)); margin: 24px auto 42px; }}
    .hero, .section, .overview {{
      background: rgba(255, 250, 241, 0.95);
      border: 1px solid rgba(143, 59, 27, 0.12);
      border-radius: 24px;
      box-shadow: 0 16px 36px rgba(80, 54, 24, 0.08);
    }}
    .hero {{ padding: 28px; }}
    .tag {{
      display: inline-block;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(143, 59, 27, 0.1);
      color: var(--accent);
      font-size: 13px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 14px 0 10px;
      font-size: clamp(34px, 5vw, 56px);
      line-height: 0.96;
      letter-spacing: -0.03em;
    }}
    .hero p {{ margin: 0; max-width: 840px; color: var(--muted); font-size: 18px; line-height: 1.6; }}
    .nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .nav a {{
      color: inherit;
      text-decoration: none;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.8);
      border: 1px solid rgba(143, 59, 27, 0.12);
      font-size: 14px;
    }}
    .overview, .section {{ margin-top: 20px; padding: 22px; }}
    h2 {{ margin: 0; font-size: 30px; letter-spacing: -0.03em; }}
    h3 {{ margin: 0 0 10px; font-size: 17px; }}
    .subtitle {{ margin: 8px 0 0; color: var(--muted); font-size: 16px; }}
    .copy {{ margin: 16px 0 0; line-height: 1.7; font-size: 16px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ padding: 14px 16px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: rgba(143, 59, 27, 0.08); font-size: 14px; text-transform: uppercase; letter-spacing: 0.04em; }}
    td {{ line-height: 1.6; font-size: 15px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid rgba(143, 59, 27, 0.11);
      border-radius: 18px;
      padding: 16px;
    }}
    .card-k {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .card-v {{ margin-top: 6px; font-size: 28px; line-height: 1.1; }}
    .card-h {{ margin-top: 8px; color: var(--muted); font-size: 14px; line-height: 1.5; }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-top: 20px;
    }}
    .chart {{
      background: var(--card);
      border: 1px solid rgba(143, 59, 27, 0.11);
      border-radius: 18px;
      padding: 16px;
    }}
    .chart-svg {{ width: 100%; height: auto; display: block; }}
    .grid {{ stroke: rgba(100, 116, 139, 0.16); stroke-width: 1; }}
    .axis-line {{ stroke: rgba(31, 41, 55, 0.62); stroke-width: 1.2; }}
    .axis, .legend {{
      fill: #475569;
      font-size: 12px;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }}
    .label {{
      fill: #334155;
      font-size: 13px;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }}
    .heat {{
      fill: #0f172a;
      font-size: 11px;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      font-weight: 600;
    }}
    .footer {{ margin-top: 14px; color: var(--muted); font-size: 14px; text-align: right; }}
    @media (max-width: 760px) {{
      .page {{ width: min(100vw - 16px, 1220px); }}
      .hero, .section, .overview {{ padding: 18px; border-radius: 18px; }}
      h1 {{ font-size: 34px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="tag">Photonic Sim / Experiments</div>
      <h1>Experiment Visual Report</h1>
      <p>这份报告把 <code>experiments/</code> 中四个 <code>run_*.py</code> 的测试目的和默认样例输出放在同一页，方便直接判断每个实验到底在测什么。</p>
      <nav class="nav">
        <a href="#step-response">Step Response</a>
        <a href="#crosstalk-scan">Crosstalk Scan</a>
        <a href="#observation-chain-sweep">Observation Chain Sweep</a>
        <a href="#drift-observation-dataset">Drift Observation Dataset</a>
      </nav>
    </header>
    <section class="overview">
      <h2>四个 Run 文件分别对应什么测试</h2>
      <table>
        <thead><tr><th>脚本</th><th>测试类型</th><th>核心问题</th></tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </section>
    {sections}
    <div class="footer">Generated from CSV files in {esc(outputs_dir)}</div>
  </div>
</body>
</html>
"""


def main():
    args = parse_args()
    outputs_dir = ROOT / args.outputs_dir
    output_path = ROOT / args.output
    expected = [
        outputs_dir / "step_response.csv",
        outputs_dir / "crosstalk_scan.csv",
        outputs_dir / "observation_chain" / "pd_sweep.csv",
        outputs_dir / "observation_chain" / "osa_sweep.csv",
        outputs_dir / "drift_dataset" / "latent_state.csv",
        outputs_dir / "drift_dataset" / "pd_frames.csv",
        outputs_dir / "drift_dataset" / "osa_frames.csv",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment outputs:\n" + "\n".join(str(path) for path in missing))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(outputs_dir), encoding="utf-8")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
