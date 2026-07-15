"""Self-contained interactive dashboard: docs/interactive/index.html.

Reuses bench/plots/make_benchmark_plots.py's MANIFEST, loaders (load_rows,
merge_rows, math500_pct, sweep_value_at, sweep_peak, _ratio_rows, _bucket_mid)
and its shared data-assembly functions (f1_panels, f2_panels, f3_points,
f4_data, f5_curve_data) verbatim, via import -- this file adds NO new raw
parsing and no new copy for anything the static charts already say. Those
five *_panels/*_points/*_data functions are themselves the single source of
truth the static SVGs (make_benchmark_plots.py) ALSO render from, so the two
can never quietly disagree about what a given panel/point/bar contains.

The only new content here is UI chrome the static SVGs never needed --
toggle buttons, section headings, footer -- defined in EXTRA_LABELS below and
merged with the imported LABELS/ROLE_LABEL/SHORT_LABEL dicts (reused, not
copied) into the JS-side lookups.

Chart library: Apache ECharts 6.1.0 (Apache-2.0), vendored at
bench/plots/vendor/echarts.min.js (see NOTICE) and inlined whole into the
output HTML -- no CDN, so the page works offline / from a bare git clone.

Determinism: no wall-clock timestamps anywhere in the output; every JSON
blob is serialized with sort_keys=True. Every sequence that carries a
meaningful order (panel order, slot order, series order, legend order) is
built from an explicit Python list, never dict/set iteration, so key-sorting
the JSON for byte-stability can never scramble it -- same discipline
make_benchmark_plots.py's own docstring documents for the static SVGs.

Usage:
  python bench/plots/make_interactive.py

Run twice and diff docs/interactive/index.html to confirm byte-identical
output.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import make_benchmark_plots as mbp  # noqa: E402  (needs sys.path insert above)

REPO_ROOT = mbp.REPO_ROOT
OUT_DIR = os.path.join(REPO_ROOT, "docs", "interactive")
OUT_PATH = os.path.join(OUT_DIR, "index.html")
VENDOR_ECHARTS = os.path.join(HERE, "vendor", "echarts.min.js")
RESULTS_RELDIR = "bench/results/"
GENERATOR_RELPATH = "bench/plots/make_interactive.py"
REGEN_CMD = "python bench/plots/make_interactive.py"

LANGS = mbp.LANGS  # ["en", "zh"] -- reused, not re-declared

# ---------------------------------------------------------------------------
# EXTRA_LABELS -- UI chrome strings the static SVGs never needed (dashboard
# nav/toggles/footer/tooltip chrome). Same shape as mbp.LABELS (id -> {en,zh})
# so both dicts merge into one flat lookup on the JS side.
# ---------------------------------------------------------------------------
EXTRA_LABELS = {
    "page_title": {"en": "rwkv-sglang benchmark dashboard", "zh": "rwkv-sglang 基准仪表盘"},
    "page_subtitle": {
        "en": "Interactive companion to the static figures in docs/BENCHMARKS.md",
        "zh": "docs/BENCHMARKS.md 中静态图表的交互版",
    },
    "lang_toggle_en": {"en": "EN", "zh": "EN"},
    "lang_toggle_zh": {"en": "中文", "zh": "中文"},
    "view_absolute": {"en": "Absolute", "zh": "绝对值"},
    "view_ratio": {"en": "Ratio vs fp16", "zh": "相对 fp16"},
    "section_f1": {"en": "F1 — RTX 5090 concurrency sweep", "zh": "F1 — RTX 5090 并发扫描"},
    "section_f2": {"en": "F2 — RTX 3090 concurrency sweep", "zh": "F2 — RTX 3090 并发扫描"},
    "section_f3": {"en": "F3 — accuracy vs. speed frontier", "zh": "F3 — 精度-速度前沿"},
    "section_f4": {"en": "F4 — MATH500 accuracy ladder", "zh": "F4 — MATH500 精度阶梯"},
    "section_f5": {"en": "F5 — positional compression", "zh": "F5 — 位置维度压缩"},
    "section_f11": {"en": "F11 — RWKV-7 vs Qwen3.5 (three readings)", "zh": "F11 — RWKV-7 对 Qwen3.5(三种读数)"},
    "section_f15": {"en": "F15 — quantization tradeoff", "zh": "F15 — 量化权衡"},
    "f5_unavailable": {
        "en": "Not yet landed — see MANIFEST key f5_positional_compression_state_precision "
              "in make_benchmark_plots.py.",
        "zh": "尚未落地 — 见 make_benchmark_plots.py 中 MANIFEST 键 "
              "f5_positional_compression_state_precision。",
    },
    "footer_data": {"en": "Data", "zh": "数据"},
    "footer_regenerate": {"en": "Regenerate", "zh": "重新生成"},
    "footer_generator": {"en": "Generator", "zh": "生成器"},
    "footer_static": {"en": "Static SVGs", "zh": "静态 SVG"},
    "tooltip_correct_of": {"en": "correct", "zh": "正确数"},
    "math500_pct_short": {"en": "MATH500 avg@64", "zh": "MATH500 avg@64"},
    "speed_short": {"en": "speed", "zh": "速度"},
    "hint_legend_click": {
        "en": "Hover for values · click a legend entry to toggle it · drag to zoom",
        "zh": "悬停查看数值 · 点击图例项可开关 "
              "· 拖拽可缩放",
    },
}

for _k, _v in EXTRA_LABELS.items():
    assert _k not in mbp.LABELS, f"EXTRA_LABELS key {_k!r} collides with mbp.LABELS"
ALL_LABELS = {**mbp.LABELS, **EXTRA_LABELS}


# ---------------------------------------------------------------------------
# Data assembly -- thin JSON-shaping over mbp's shared *_panels/*_points/
# *_data functions. No raw file is opened here; every number below traces
# back to one of those shared functions (or, transitively, to load_rows /
# merge_rows / math500_pct / sweep_value_at / sweep_peak inside them).
# ---------------------------------------------------------------------------
def _panel_series_json(series_tuples, fp16_rows):
    """[(rows, role, connect[, hollow, linestyle, label_key])...] -> JSON-able
    list of dicts carrying both the absolute rows and the (shared-x-only,
    Phase 2a) ratio-vs-fp16 rows for each series -- computed via mbp._ratio_rows,
    the exact same function the static SVG ratio sub-panels use.
    """
    out = []
    for tup in series_tuples:
        if len(tup) == 3:
            rows, role, connect = tup
            hollow, linestyle, label_key = False, "-", None
        else:
            rows, role, connect, hollow, linestyle, label_key = tup
        out.append({
            "role": role,
            "connect": connect,
            "hollow": hollow,
            "linestyle": linestyle,
            "label_key": label_key,
            "abs": rows,
            "ratio": None if role == "fp16" else mbp._ratio_rows(rows, fp16_rows),
        })
    return out


def build_f1_json():
    panels = []
    for p in mbp.f1_panels():
        fp16_rows = p["series"][0][0]
        panels.append({"title": p["title"], "series": _panel_series_json(p["series"], fp16_rows)})
    return panels


def build_f2_json():
    panels = []
    for p in mbp.f2_panels():
        fp16_rows = p["series"][0][0]
        entry = {"series": _panel_series_json(p["series"], fp16_rows)}
        if "title" in p:
            entry["title"] = p["title"]
        else:
            entry["title_key"] = p["title_key"]
        if "cliff" in p:
            entry["cliff"] = p["cliff"]
        panels.append(entry)
    return panels


def build_f3_json():
    # f3_points() already returns JSON-able dicts (see make_benchmark_plots.py) --
    # passed through unchanged, no reshaping needed.
    return mbp.f3_points()


def build_f4_json():
    d = mbp.f4_data()

    def pack(group):
        return {role: {"pct": pct, "correct": c, "total": n} for role, (pct, c, n) in group.items()}

    return {
        "slots": d["slots"],
        "slot_label": d["slot_label"],
        "onepfive": pack(d["onepfive"]),
        "seven2b": pack(d["seven2b"]),
    }


def build_f5_json():
    cd = mbp.f5_curve_data()
    if cd is None:
        return None
    return {
        "bucket_ticks": cd["bucket_ticks"],
        "rows_by_role": cd["rows_by_role"],
        "delta": cd["delta"],
        "noise": cd["noise"],
    }


def build_f11_json():
    # mbp.f11_data() -> per-tier bsz1 bars + (peak, concurrency) pairs. Reshaped
    # to JSON (tuples -> lists); every number traces back to load_rows/sweep_peak.
    return [{
        "size": t["size"],
        "rwkv_deploy": t["rwkv_deploy"],
        "rwkv_bf16": t["rwkv_bf16"],
        "qwen_bf16": t["qwen_bf16"],
        "rwkv_peak": list(t["rwkv_peak"]),
        "qwen_peak": list(t["qwen_peak"]),
    } for t in mbp.f11_data()]


def build_f15_json():
    panel_a = [{"size_key": sk, "tiers": [{"role": r, "delta": d} for r, d in tiers]}
               for sk, tiers in mbp.f15a_data()]
    return {"panelA": panel_a, "panelB": mbp.f15b_data()}


def build_data():
    return {
        "f1": build_f1_json(),
        "f2": build_f2_json(),
        "f3": build_f3_json(),
        "f4": build_f4_json(),
        "f5": build_f5_json(),
        "f11": build_f11_json(),
        "f15": build_f15_json(),
    }


# ---------------------------------------------------------------------------
# CSS -- reuses the exact palette constants make_benchmark_plots.py already
# validated (the dataviz skill's categorical instance; see that module's own
# HUE/INK/... comments), so the interactive dashboard and the static SVGs
# read as one system rather than two differently-themed artifacts.
# ---------------------------------------------------------------------------
def build_css():
    return f"""
:root {{
  --ink: {mbp.INK};
  --ink-secondary: {mbp.INK_SECONDARY};
  --ink-muted: {mbp.INK_MUTED};
  --grid: {mbp.GRID};
  --axis: {mbp.AXIS};
  --surface: {mbp.SURFACE};
  --alert: {mbp.ALERT};
  --card: #ffffff;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans SC",
          "Microsoft YaHei", "Hiragino Sans GB", Roboto, Helvetica, Arial, sans-serif;
}}
* {{ box-sizing: border-box; }}
html, body {{
  margin: 0; padding: 0; background: var(--surface); color: var(--ink);
  font-family: var(--font); -webkit-font-smoothing: antialiased;
}}
body {{ padding-bottom: 48px; }}
a {{ color: {mbp.HUE["blue"]}; }}
header.dash-header {{
  display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap;
  gap: 12px; padding: 22px 28px 16px; border-bottom: 1px solid var(--grid);
}}
header.dash-header .titles h1 {{ font-size: 20px; margin: 0 0 4px; font-weight: 650; }}
header.dash-header .titles p {{ font-size: 13px; margin: 0; color: var(--ink-muted); }}
.lang-toggle {{ display: flex; gap: 6px; }}
.btn {{
  font-family: var(--font); font-size: 12.5px; padding: 6px 12px; border-radius: 999px;
  border: 1px solid var(--axis); background: var(--card); color: var(--ink-secondary);
  cursor: pointer; line-height: 1.2;
}}
.btn:hover {{ border-color: {mbp.HUE["blue"]}; color: {mbp.HUE["blue"]}; }}
.btn.active {{ background: var(--ink); color: var(--surface); border-color: var(--ink); }}
main {{ max-width: 1360px; margin: 0 auto; padding: 0 28px; }}
section.chart-section {{
  margin-top: 34px; padding-top: 18px; border-top: 1px solid var(--grid);
}}
section.chart-section:first-of-type {{ border-top: none; }}
.section-head {{
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;
  margin-bottom: 6px;
}}
.section-head h2 {{ font-size: 16px; margin: 0; font-weight: 620; }}
.view-toggle {{ display: flex; gap: 6px; }}
.hint {{ font-size: 11.5px; color: var(--ink-muted); margin: 0 0 12px; font-style: italic; }}
.panel-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.panel-card {{
  flex: 1 1 320px; min-width: 300px; background: var(--card); border: 1px solid var(--grid);
  border-radius: 10px; padding: 6px;
}}
.chart-el {{ width: 100%; height: 380px; }}
.chart-el.tall {{ height: 460px; }}
.chart-el.wide {{ height: 460px; }}
.single-card {{ background: var(--card); border: 1px solid var(--grid); border-radius: 10px; padding: 6px; }}
.unavailable {{
  padding: 28px; text-align: center; color: var(--ink-muted); font-size: 13px; font-style: italic;
  background: var(--card); border: 1px dashed var(--axis); border-radius: 10px;
}}
footer.dash-footer {{
  max-width: 1360px; margin: 40px auto 0; padding: 16px 28px; border-top: 1px solid var(--grid);
  font-size: 11.5px; color: var(--ink-muted); display: flex; gap: 22px; flex-wrap: wrap;
}}
footer.dash-footer code {{
  background: var(--grid); padding: 1px 5px; border-radius: 4px; font-size: 11px;
}}
.echarts-tooltip-note {{ color: var(--ink-muted); font-size: 11px; }}
""".strip("\n")


# ---------------------------------------------------------------------------
# DASHBOARD_JS -- plain JS, no Python templating inside it (this whole string
# is injected as one opaque block via a sentinel-token .replace(), so any {}/
# `${}` it contains is never re-parsed by Python -- see build_html()).
# ---------------------------------------------------------------------------
DASHBOARD_JS = r"""
(function () {
  "use strict";
  var DATA = window.__DASH_DATA__;
  var LABELS = window.__DASH_LABELS__;
  var ROLE_LABEL = window.__DASH_ROLE_LABEL__;
  var SHORT_LABEL = window.__DASH_SHORT_LABEL__;
  var ROLE_COLOR = window.__DASH_ROLE_COLOR__;
  var XCOLOR = window.__DASH_XCOLOR__;
  var THEME = window.__DASH_THEME__;

  var state = { lang: "en", f1view: "abs", f2view: "abs" };
  var chartInstances = {};

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function t(key) {
    var e = LABELS[key];
    if (!e) return key;
    return e[state.lang] || e.en || key;
  }
  function rl(role) {
    var e = ROLE_LABEL[role];
    if (!e) return role;
    return e[state.lang] || e.en || role;
  }
  function fmtInt(v) {
    if (v == null || isNaN(v)) return "";
    return Number(v).toLocaleString(state.lang === "zh" ? "zh-CN" : "en-US", { maximumFractionDigits: 0 });
  }
  function fmtRatioTick(v) {
    return v.toFixed(2) + "×";
  }

  function applyI18nText() {
    document.title = t("page_title");
    var nodes = document.querySelectorAll("[data-i18n]");
    for (var i = 0; i < nodes.length; i++) {
      nodes[i].textContent = t(nodes[i].getAttribute("data-i18n"));
    }
    document.documentElement.setAttribute("lang", state.lang === "zh" ? "zh-CN" : "en");
  }

  function setActiveButtons() {
    document.getElementById("lang-btn-en").classList.toggle("active", state.lang === "en");
    document.getElementById("lang-btn-zh").classList.toggle("active", state.lang === "zh");
    document.getElementById("f1-btn-abs").classList.toggle("active", state.f1view === "abs");
    document.getElementById("f1-btn-ratio").classList.toggle("active", state.f1view === "ratio");
    document.getElementById("f2-btn-abs").classList.toggle("active", state.f2view === "abs");
    document.getElementById("f2-btn-ratio").classList.toggle("active", state.f2view === "ratio");
  }

  function getChart(id) {
    if (!chartInstances[id]) {
      chartInstances[id] = echarts.init(document.getElementById(id), null, { renderer: "svg" });
    }
    return chartInstances[id];
  }

  function logXAxis(showLabel, nameKey) {
    return {
      type: "log",
      logBase: 2,
      name: showLabel ? t(nameKey || "concurrency") : "",
      nameLocation: "middle",
      nameGap: 20,
      nameTextStyle: { color: THEME.ink_secondary, fontSize: 11.5 },
      axisLabel: { show: showLabel !== false, formatter: fmtInt, color: THEME.ink_muted, fontSize: 10.5 },
      axisLine: { lineStyle: { color: THEME.axis } },
      splitLine: { lineStyle: { color: THEME.grid } },
    };
  }
  // concurrency sweeps (F1/F2) are the common case -- keep the short name used
  // at every existing call site; F5's bucket-position axis passes its own key.
  function concurrencyXAxis(showLabel) {
    return logXAxis(showLabel, "concurrency");
  }

  // F5's main panel: bits/token in a ~0-4 range, where fmtInt's integer
  // rounding collapsed distinct neighboring ticks (e.g. 3.5 and 3.0 both ->
  // "3") into visually duplicate labels -- one decimal place fixes that.
  function fmtDecimal1(v) {
    return Number(v).toFixed(1);
  }
  // F5's delta panel: tiny (~1e-4) values where fmtInt rounds everything to
  // "0" -- mirror make_benchmark_plots.py's own _delta_fmt (short scientific
  // form, "0" spelled out exactly, matching the static SVG's delta sub-panel).
  function fmtDelta(v) {
    if (v === 0) return "0";
    var s = v.toExponential(0); // e.g. "-1e-4", "5e-5"
    return s.replace("e-0", "e-").replace("e+0", "e");
  }

  // Legend anchored BELOW the title, wrapping (type:"plain") rather than the
  // single-line paginated "scroll" type -- these panels are narrow (as low as
  // ~300px) with up to 5 long labels, and a bottom-anchored legend was found
  // (see task notes / screenshot) to collide with the x-axis name + dataZoom
  // slider once both had to share the same bottom strip. Top placement gives
  // it its own reserved band (grid.top) with nothing else contending for it.
  function topLegend(legendNames) {
    return {
      data: legendNames, top: 26, left: 4, right: 4, type: "plain",
      itemGap: 10, itemWidth: 15, itemHeight: 10,
      textStyle: { color: THEME.ink_secondary, fontSize: 10.5 },
    };
  }

  function valueYAxis(nameKey, opts) {
    opts = opts || {};
    return Object.assign({
      type: "value",
      name: nameKey ? t(nameKey) : "",
      nameLocation: "middle",
      nameGap: opts.nameGap || 52,
      nameTextStyle: { color: THEME.ink_secondary, fontSize: 11.5 },
      axisLabel: { formatter: opts.formatter || fmtInt, color: THEME.ink_muted },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: THEME.grid } },
    }, opts.extra || {});
  }

  function makeLineSeries(role, points, label, connect, hollow, linestyle) {
    var color = ROLE_COLOR[role] || THEME.ink;
    var dashed = linestyle === "--";
    return {
      name: label,
      type: "line",
      data: points || [],
      showSymbol: true,
      symbol: hollow ? "diamond" : "circle",
      symbolSize: hollow ? 11 : (points && points.length <= 1 ? 13 : 7),
      connectNulls: false,
      lineStyle: { width: 2, color: color, type: dashed ? "dashed" : "solid" },
      itemStyle: {
        color: hollow ? THEME.surface : color,
        borderColor: color,
        borderWidth: hollow ? 2.2 : 1.4,
      },
      emphasis: { focus: "series" },
      z: 3,
    };
  }

  function fp16ReferenceSeries() {
    return {
      name: rl("fp16"),
      type: "line",
      data: [],
      silent: true,
      tooltip: { show: false },
      markLine: {
        symbol: "none",
        silent: true,
        data: [{ yAxis: 1 }],
        lineStyle: { color: ROLE_COLOR.fp16, type: "dashed", width: 1.5 },
        label: { formatter: "1×", color: ROLE_COLOR.fp16, position: "insideEndTop", fontSize: 10 },
      },
    };
  }

  function lineTooltip() {
    return {
      trigger: "axis",
      axisPointer: { type: "cross", label: { backgroundColor: THEME.ink_secondary } },
      valueFormatter: function (v) {
        return typeof v === "number" ? fmtInt(v) : v;
      },
    };
  }

  function baseDataZoom(xAxisIndex) {
    var idx = xAxisIndex === undefined ? 0 : xAxisIndex;
    return [
      { type: "inside", xAxisIndex: idx },
      { type: "slider", xAxisIndex: idx, height: 10, bottom: 4, borderColor: THEME.axis, fillerColor: "rgba(42,120,214,0.12)" },
    ];
  }

  // ---- F1 -------------------------------------------------------------
  function renderF1() {
    var panels = DATA.f1;
    var view = state.f1view;
    for (var i = 0; i < panels.length; i++) {
      var panel = panels[i];
      var chart = getChart("f1-panel-" + i);
      var series = [];
      var legendNames = [];
      for (var j = 0; j < panel.series.length; j++) {
        var s = panel.series[j];
        var label = s.label_key ? t(s.label_key) : rl(s.role);
        if (view === "abs") {
          series.push(makeLineSeries(s.role, s.abs, label, s.connect, s.hollow, s.linestyle));
          legendNames.push(label);
        } else if (s.role !== "fp16") {
          series.push(makeLineSeries(s.role, s.ratio, label, s.connect, s.hollow, s.linestyle));
          legendNames.push(label);
        }
      }
      if (view === "ratio") {
        legendNames.push(rl("fp16"));
        series.push(fp16ReferenceSeries());
      }
      chart.setOption({
        backgroundColor: "transparent",
        title: { text: panel.title, left: 6, top: 2, textStyle: { fontSize: 13.5, color: THEME.ink, fontWeight: 600 } },
        grid: { left: 58, right: 20, top: 92, bottom: 62 },
        tooltip: lineTooltip(),
        legend: topLegend(legendNames),
        xAxis: concurrencyXAxis(true),
        yAxis: view === "abs" ? valueYAxis("tok_s") : valueYAxis("ratio_vs_fp16", { formatter: fmtRatioTick }),
        dataZoom: baseDataZoom(0),
        series: series,
      }, true);
    }
  }

  // ---- F2 -------------------------------------------------------------
  function renderF2() {
    var panels = DATA.f2;
    var view = state.f2view;
    for (var i = 0; i < panels.length; i++) {
      var panel = panels[i];
      var chart = getChart("f2-panel-" + i);
      var series = [];
      var legendNames = [];
      for (var j = 0; j < panel.series.length; j++) {
        var s = panel.series[j];
        var label = s.label_key ? t(s.label_key) : rl(s.role);
        if (view === "abs") {
          series.push(makeLineSeries(s.role, s.abs, label, s.connect, s.hollow, s.linestyle));
          legendNames.push(label);
        } else if (s.role !== "fp16") {
          series.push(makeLineSeries(s.role, s.ratio, label, s.connect, s.hollow, s.linestyle));
          legendNames.push(label);
        }
      }
      if (view === "ratio") {
        legendNames.push(rl("fp16"));
        series.push(fp16ReferenceSeries());
      }
      var title = panel.title || t(panel.title_key);
      chart.setOption({
        backgroundColor: "transparent",
        title: { text: title, left: 6, top: 2, textStyle: { fontSize: 13.5, color: THEME.ink, fontWeight: 600 } },
        grid: { left: 58, right: 20, top: 92, bottom: 62 },
        tooltip: lineTooltip(),
        legend: topLegend(legendNames),
        xAxis: concurrencyXAxis(true),
        yAxis: view === "abs" ? valueYAxis("tok_s") : valueYAxis("ratio_vs_fp16", { formatter: fmtRatioTick }),
        dataZoom: baseDataZoom(0),
        series: series,
      }, true);
    }
  }

  // ---- F3 -- accuracy/speed frontier scatter --------------------------
  function renderF3() {
    var chart = getChart("f3-chart");
    var points = DATA.f3;
    var series = [];
    var legendNames = [];
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      var label = t(p.label_key);
      legendNames.push(label);
      series.push({
        name: label,
        type: "scatter",
        symbol: p.hollow ? "diamond" : "circle",
        symbolSize: p.hollow ? 20 : 17,
        itemStyle: {
          color: p.hollow ? THEME.surface : ROLE_COLOR[p.role],
          borderColor: ROLE_COLOR[p.role],
          borderWidth: p.hollow ? 2.4 : 1.3,
        },
        data: [{
          value: [p.x, p.y],
          noteKey: p.note_key,
          correct: p.correct,
          total: p.total,
          redGate: !!p.red_gate,
          labelText: label,
        }],
      });
    }
    chart.setOption({
      backgroundColor: "transparent",
      title: { text: t("f3_title"), left: 6, top: 2, textStyle: { fontSize: 14, color: THEME.ink, fontWeight: 600 } },
      grid: { left: 74, right: 40, top: 64, bottom: 66 },
      tooltip: {
        trigger: "item",
        formatter: function (pt) {
          var d = pt.data;
          var html = "<strong>" + escapeHtml(d.labelText) + "</strong><br/>"
            + escapeHtml(t("speed_short")) + ": " + fmtInt(d.value[0]) + " tok/s<br/>"
            + escapeHtml(t("math500_pct_short")) + ": " + d.value[1].toFixed(2) + "% ("
            + d.correct + "/" + d.total + " " + escapeHtml(t("tooltip_correct_of")) + ")<br/>"
            + '<span class="echarts-tooltip-note">' + escapeHtml(t(d.noteKey)) + "</span>";
          if (d.redGate) {
            html += '<br/><strong style="color:' + THEME.alert + '">' + escapeHtml(t("f3_red_gate")) + "</strong>";
          }
          return html;
        },
      },
      legend: Object.assign(topLegend(legendNames), { top: 28 }),
      xAxis: Object.assign({}, valueYAxis(null), {
        type: "value", name: t("f3_xlabel"), nameLocation: "middle", nameGap: 20,
        axisLabel: { formatter: fmtInt, color: THEME.ink_muted },
      }),
      yAxis: valueYAxis("math500_ylabel", { nameGap: 48 }),
      dataZoom: baseDataZoom(0),
      series: series,
    }, true);
  }

  // ---- F4 -- MATH500 ladder (grouped bars) -----------------------------
  function renderF4() {
    var chart = getChart("f4-chart");
    var d = DATA.f4;
    var categories = ["1.5B", "7.2B"];
    var groups = [d.onepfive, d.seven2b];
    var series = d.slots.map(function (slot) {
      return {
        name: d.slot_label[slot],
        type: "bar",
        barMaxWidth: 26,
        itemStyle: { color: ROLE_COLOR[slot], borderRadius: [3, 3, 0, 0] },
        data: categories.map(function (cat, ci) {
          var entry = groups[ci][slot];
          return entry ? { value: entry.pct, correct: entry.correct, total: entry.total } : null;
        }),
        label: {
          show: true, position: "top", fontSize: 9.5, color: THEME.ink,
          formatter: function (p) { return p.value == null ? "" : p.value.toFixed(1); },
        },
      };
    });
    chart.setOption({
      backgroundColor: "transparent",
      title: { text: t("f4_title"), left: 6, top: 2, textStyle: { fontSize: 14, color: THEME.ink, fontWeight: 600 } },
      grid: { left: 60, right: 20, top: 74, bottom: 44 },
      tooltip: {
        trigger: "item",
        formatter: function (p) {
          if (!p.data) return "";
          return "<strong>" + escapeHtml(p.seriesName) + "</strong> · " + escapeHtml(p.name) + "<br/>"
            + p.data.value.toFixed(2) + "% (" + p.data.correct + "/" + p.data.total + " "
            + escapeHtml(t("tooltip_correct_of")) + ")";
        },
      },
      legend: Object.assign(topLegend(d.slots.map(function (s) { return d.slot_label[s]; })), { top: 28 }),
      xAxis: { type: "category", data: categories, axisLabel: { color: THEME.ink, fontSize: 13 }, axisLine: { lineStyle: { color: THEME.axis } }, axisTick: { show: false } },
      yAxis: valueYAxis("math500_ylabel", { extra: { max: 75 } }),
      series: series,
    }, true);
  }

  // ---- F5 -- positional compression (dual grid: curve + delta) --------
  function renderF5() {
    var section = document.getElementById("f5-available");
    var unavail = document.getElementById("f5-unavailable-box");
    if (!DATA.f5) {
      section.style.display = "none";
      unavail.style.display = "block";
      return;
    }
    section.style.display = "block";
    unavail.style.display = "none";

    var chart = getChart("f5-chart");
    var d = DATA.f5;
    var rowsOff = d.rows_by_role.fp16;
    var rowsOn = d.rows_by_role.fp16_state_fp16;
    var deltaData = d.delta.x.map(function (x, i) { return [x, d.delta.y[i]]; });

    chart.setOption({
      backgroundColor: "transparent",
      title: { text: t("f5_title"), left: 6, top: 2, textStyle: { fontSize: 14, color: THEME.ink, fontWeight: 600 } },
      grid: [
        { left: 76, right: 30, top: 50, height: "46%" },
        { left: 76, right: 30, top: "66%", height: "18%" },
      ],
      tooltip: lineTooltip(),
      legend: { data: [t("f5_pt_off"), t("f5_pt_on")], top: 26, textStyle: { color: THEME.ink_secondary, fontSize: 10.5 } },
      xAxis: [
        Object.assign(logXAxis(false, "f5_xlabel"), { gridIndex: 0 }),
        Object.assign(logXAxis(true, "f5_xlabel"), { gridIndex: 1 }),
      ],
      yAxis: [
        Object.assign(valueYAxis("f5_ylabel", { nameGap: 58, formatter: fmtDecimal1 }), { gridIndex: 0 }),
        Object.assign(valueYAxis("f5_delta_ylabel", { nameGap: 58, formatter: fmtDelta, extra: { min: -1e-4, max: 1e-4 } }), { gridIndex: 1 }),
      ],
      series: [
        Object.assign(makeLineSeries("fp16", rowsOff, t("f5_pt_off"), true, false, "-"), { xAxisIndex: 0, yAxisIndex: 0 }),
        Object.assign(makeLineSeries("fp16_state_fp16", rowsOn, t("f5_pt_on"), true, false, "-"), { xAxisIndex: 0, yAxisIndex: 0 }),
        {
          name: t("f5_delta_title"), type: "line", xAxisIndex: 1, yAxisIndex: 1, data: deltaData,
          showSymbol: true, symbolSize: 5, lineStyle: { color: THEME.ink, width: 1.4 },
          itemStyle: { color: THEME.ink }, z: 3,
          markArea: {
            silent: true, itemStyle: { color: THEME.noise_band, opacity: 0.6 },
            data: [[{ yAxis: -d.noise }, { yAxis: d.noise }]],
          },
          markLine: { silent: true, symbol: "none", lineStyle: { color: THEME.axis, width: 1 }, data: [{ yAxis: 0 }] },
        },
      ],
      dataZoom: baseDataZoom([0, 1]),
    }, true);
  }

  // ---- F11 -- RWKV-7 vs Qwen3.5, three readings ----------------------
  function catAxis(cats) {
    return { type: "category", data: cats, axisLabel: { color: THEME.ink, fontSize: 12 },
             axisLine: { lineStyle: { color: THEME.axis } }, axisTick: { show: false } };
  }
  function barValueTooltip() {
    return { trigger: "axis", axisPointer: { type: "shadow" },
             valueFormatter: function (v) { return v == null ? "" : fmtInt(v) + " tok/s"; } };
  }
  function tokLabel(digits) {
    return { show: true, position: "top", fontSize: 9, color: THEME.ink,
             formatter: function (p) { return p.value == null ? "" : fmtInt(p.value); } };
  }
  function renderF11() {
    var tiers = DATA.f11;
    var cats = tiers.map(function (x) { return x.size; });
    getChart("f11-panel-0").setOption({
      backgroundColor: "transparent",
      title: { text: t("f11_pa_title"), left: 6, top: 2, textStyle: { fontSize: 12, color: THEME.ink, fontWeight: 600 } },
      grid: { left: 60, right: 18, top: 80, bottom: 32 },
      tooltip: barValueTooltip(),
      legend: topLegend([t("f11_rwkv_deploy"), t("f11_rwkv_bf16"), t("f11_qwen")]),
      xAxis: catAxis(cats),
      yAxis: valueYAxis("tok_s"),
      series: [
        { name: t("f11_rwkv_deploy"), type: "bar", barMaxWidth: 26, itemStyle: { color: XCOLOR.ours }, label: tokLabel(), data: tiers.map(function (x) { return x.rwkv_deploy; }) },
        { name: t("f11_rwkv_bf16"), type: "bar", barMaxWidth: 26, itemStyle: { color: XCOLOR.ours_soft }, label: tokLabel(), data: tiers.map(function (x) { return x.rwkv_bf16; }) },
        { name: t("f11_qwen"), type: "bar", barMaxWidth: 26, itemStyle: { color: XCOLOR.qwen }, label: tokLabel(), data: tiers.map(function (x) { return x.qwen_bf16; }) },
      ],
    }, true);
    getChart("f11-panel-1").setOption({
      backgroundColor: "transparent",
      title: { text: t("f11_pb_title"), left: 6, top: 2, textStyle: { fontSize: 12, color: THEME.ink, fontWeight: 600 } },
      grid: { left: 60, right: 18, top: 80, bottom: 32 },
      tooltip: barValueTooltip(),
      legend: topLegend([t("f11_rwkv_bf16"), t("f11_qwen")]),
      xAxis: catAxis(cats),
      yAxis: valueYAxis("tok_s"),
      series: [
        { name: t("f11_rwkv_bf16"), type: "bar", barMaxWidth: 34, itemStyle: { color: XCOLOR.ours_soft }, label: tokLabel(), data: tiers.map(function (x) { return x.rwkv_peak[0]; }) },
        { name: t("f11_qwen"), type: "bar", barMaxWidth: 34, itemStyle: { color: XCOLOR.qwen }, label: tokLabel(), data: tiers.map(function (x) { return x.qwen_peak[0]; }) },
      ],
    }, true);
  }

  // ---- F15 -- quantization tradeoff ---------------------------------
  function renderF15() {
    var d = DATA.f15;
    var cats = d.panelA.map(function (c) { return t(c.size_key); });
    var slots = ["w8g64", "w8a8", "int4_gptq"];
    var seriesA = slots.map(function (role) {
      return {
        name: rl(role), type: "bar", barMaxWidth: 30, itemStyle: { color: ROLE_COLOR[role] },
        data: d.panelA.map(function (c) {
          var found = null;
          for (var i = 0; i < c.tiers.length; i++) { if (c.tiers[i].role === role) found = c.tiers[i].delta; }
          return found;
        }),
        label: { show: true, position: "top", fontSize: 8.5, color: THEME.ink,
                 formatter: function (p) { return p.value == null ? "" : "+" + Number(p.value).toFixed(4); } },
      };
    });
    getChart("f15-panel-0").setOption({
      backgroundColor: "transparent",
      title: { text: t("f15a_title"), left: 6, top: 2, textStyle: { fontSize: 11, color: THEME.ink, fontWeight: 600 } },
      grid: { left: 64, right: 18, top: 76, bottom: 32 },
      tooltip: { trigger: "item", formatter: function (p) { return "<strong>" + escapeHtml(p.seriesName) + "</strong> · " + escapeHtml(p.name) + "<br/>Δ +" + Number(p.value).toFixed(4) + " bpb"; } },
      legend: topLegend(slots.map(function (s) { return rl(s); })),
      xAxis: catAxis(cats),
      yAxis: valueYAxis("f15a_ylabel", { nameGap: 60, formatter: function (v) { return Number(v).toFixed(3); } }),
      series: seriesA,
    }, true);
    var seriesB = d.panelB.map(function (p) {
      return {
        name: rl(p.role), type: "scatter", symbolSize: Math.sqrt(p.gb) * 20,
        itemStyle: { color: ROLE_COLOR[p.role], borderColor: THEME.surface, borderWidth: 1.3 },
        data: [{ value: [p.speed, p.acc], gb: p.gb, cross: p.cross, labelText: rl(p.role) }],
      };
    });
    getChart("f15-panel-1").setOption({
      backgroundColor: "transparent",
      title: { text: t("f15b_title"), left: 6, top: 2, textStyle: { fontSize: 11, color: THEME.ink, fontWeight: 600 } },
      grid: { left: 56, right: 26, top: 76, bottom: 46 },
      tooltip: { trigger: "item", formatter: function (pt) {
        var dd = pt.data;
        var html = "<strong>" + escapeHtml(dd.labelText) + "</strong><br/>"
          + escapeHtml(t("speed_short")) + ": " + fmtInt(dd.value[0]) + " tok/s<br/>"
          + escapeHtml(t("math500_pct_short")) + ": " + dd.value[1].toFixed(1) + "%<br/>~"
          + Number(dd.gb).toFixed(1) + " GB";
        if (dd.cross) { html += '<br/><span class="echarts-tooltip-note">' + escapeHtml(t("f15b_cross")) + "</span>"; }
        return html;
      } },
      legend: topLegend(d.panelB.map(function (p) { return rl(p.role); })),
      xAxis: Object.assign({}, valueYAxis(null), { type: "value", name: t("f15b_xlabel"), nameLocation: "middle", nameGap: 22, axisLabel: { formatter: fmtInt, color: THEME.ink_muted } }),
      yAxis: valueYAxis("math500_ylabel", { nameGap: 40 }),
      series: seriesB,
    }, true);
  }

  function renderAll() {
    applyI18nText();
    setActiveButtons();
    renderF1();
    renderF2();
    renderF3();
    renderF4();
    renderF5();
    renderF11();
    renderF15();
  }

  function wireControls() {
    document.getElementById("lang-btn-en").addEventListener("click", function () { state.lang = "en"; renderAll(); });
    document.getElementById("lang-btn-zh").addEventListener("click", function () { state.lang = "zh"; renderAll(); });
    document.getElementById("f1-btn-abs").addEventListener("click", function () { state.f1view = "abs"; setActiveButtons(); renderF1(); });
    document.getElementById("f1-btn-ratio").addEventListener("click", function () { state.f1view = "ratio"; setActiveButtons(); renderF1(); });
    document.getElementById("f2-btn-abs").addEventListener("click", function () { state.f2view = "abs"; setActiveButtons(); renderF2(); });
    document.getElementById("f2-btn-ratio").addEventListener("click", function () { state.f2view = "ratio"; setActiveButtons(); renderF2(); });
    window.addEventListener("resize", function () {
      for (var id in chartInstances) {
        if (Object.prototype.hasOwnProperty.call(chartInstances, id)) chartInstances[id].resize();
      }
    });
  }

  function boot() {
    wireControls();
    renderAll();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
""".strip("\n")


# ---------------------------------------------------------------------------
# HTML skeleton -- sentinel-token substitution only (no Python str.format/
# f-string scanning of the JS/CSS/JSON payloads), so nothing inside those
# payloads (JS template literals, minified-JS braces, JSON braces) can ever
# be misread as a Python format placeholder.
# ---------------------------------------------------------------------------
HTML_SKELETON = """<title>@@PAGE_TITLE@@</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<style>
@@CSS@@
</style>
<header class="dash-header">
  <div class="titles">
    <h1 data-i18n="page_title">@@PAGE_TITLE@@</h1>
    <p data-i18n="page_subtitle"></p>
  </div>
  <div class="lang-toggle" role="group" aria-label="language">
    <button id="lang-btn-en" class="btn active" type="button" data-i18n="lang_toggle_en">EN</button>
    <button id="lang-btn-zh" class="btn" type="button" data-i18n="lang_toggle_zh">中文</button>
  </div>
</header>
<main>
  <p class="hint" data-i18n="hint_legend_click"></p>

  <section class="chart-section" id="section-f1">
    <div class="section-head">
      <h2 data-i18n="section_f1"></h2>
      <div class="view-toggle" role="group" aria-label="F1 view">
        <button id="f1-btn-abs" class="btn active" type="button" data-i18n="view_absolute"></button>
        <button id="f1-btn-ratio" class="btn" type="button" data-i18n="view_ratio"></button>
      </div>
    </div>
    <div class="panel-row">
      <div class="panel-card"><div id="f1-panel-0" class="chart-el"></div></div>
      <div class="panel-card"><div id="f1-panel-1" class="chart-el"></div></div>
      <div class="panel-card"><div id="f1-panel-2" class="chart-el"></div></div>
    </div>
  </section>

  <section class="chart-section" id="section-f2">
    <div class="section-head">
      <h2 data-i18n="section_f2"></h2>
      <div class="view-toggle" role="group" aria-label="F2 view">
        <button id="f2-btn-abs" class="btn active" type="button" data-i18n="view_absolute"></button>
        <button id="f2-btn-ratio" class="btn" type="button" data-i18n="view_ratio"></button>
      </div>
    </div>
    <div class="panel-row">
      <div class="panel-card"><div id="f2-panel-0" class="chart-el"></div></div>
      <div class="panel-card"><div id="f2-panel-1" class="chart-el"></div></div>
    </div>
  </section>

  <section class="chart-section" id="section-f3">
    <div class="section-head"><h2 data-i18n="section_f3"></h2></div>
    <div class="single-card"><div id="f3-chart" class="chart-el wide"></div></div>
  </section>

  <section class="chart-section" id="section-f4">
    <div class="section-head"><h2 data-i18n="section_f4"></h2></div>
    <div class="single-card"><div id="f4-chart" class="chart-el wide"></div></div>
  </section>

  <section class="chart-section" id="section-f5">
    <div class="section-head"><h2 data-i18n="section_f5"></h2></div>
    <div id="f5-available" class="single-card"><div id="f5-chart" class="chart-el tall"></div></div>
    <div id="f5-unavailable-box" class="unavailable" data-i18n="f5_unavailable" style="display:none"></div>
  </section>

  <section class="chart-section" id="section-f11">
    <div class="section-head"><h2 data-i18n="section_f11"></h2></div>
    <div class="panel-row">
      <div class="panel-card"><div id="f11-panel-0" class="chart-el"></div></div>
      <div class="panel-card"><div id="f11-panel-1" class="chart-el"></div></div>
    </div>
  </section>

  <section class="chart-section" id="section-f15">
    <div class="section-head"><h2 data-i18n="section_f15"></h2></div>
    <div class="panel-row">
      <div class="panel-card"><div id="f15-panel-0" class="chart-el"></div></div>
      <div class="panel-card"><div id="f15-panel-1" class="chart-el"></div></div>
    </div>
  </section>
</main>

<footer class="dash-footer">
  <span data-i18n="footer_data"></span>: <code>@@RESULTS_RELDIR@@</code>
  <span data-i18n="footer_regenerate"></span>: <code>@@REGEN_CMD@@</code>
  <span data-i18n="footer_generator"></span>: <code>@@GENERATOR_RELPATH@@</code>
  <span data-i18n="footer_static"></span>: <code>docs/BENCHMARKS.md</code>
</footer>

<script>
@@ECHARTS_JS@@
</script>
<script>
window.__DASH_DATA__ = @@DATA_JSON@@;
window.__DASH_LABELS__ = @@LABELS_JSON@@;
window.__DASH_ROLE_LABEL__ = @@ROLE_LABEL_JSON@@;
window.__DASH_SHORT_LABEL__ = @@SHORT_LABEL_JSON@@;
window.__DASH_ROLE_COLOR__ = @@ROLE_COLOR_JSON@@;
window.__DASH_XCOLOR__ = @@XCOLOR_JSON@@;
window.__DASH_THEME__ = @@THEME_JSON@@;
</script>
<script>
@@DASHBOARD_JS@@
</script>
"""


def _dump(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def build_html():
    with open(VENDOR_ECHARTS, "r", encoding="utf-8") as f:
        echarts_js = f.read()

    data = build_data()
    theme = {
        "ink": mbp.INK, "ink_secondary": mbp.INK_SECONDARY, "ink_muted": mbp.INK_MUTED,
        "grid": mbp.GRID, "axis": mbp.AXIS, "surface": mbp.SURFACE, "alert": mbp.ALERT,
        "noise_band": mbp.NOISE_BAND,
    }

    html = HTML_SKELETON
    replacements = {
        "@@PAGE_TITLE@@": ALL_LABELS["page_title"]["en"],
        "@@CSS@@": build_css(),
        "@@RESULTS_RELDIR@@": RESULTS_RELDIR,
        "@@REGEN_CMD@@": REGEN_CMD,
        "@@GENERATOR_RELPATH@@": GENERATOR_RELPATH,
        "@@ECHARTS_JS@@": echarts_js,
        "@@DATA_JSON@@": _dump(data),
        "@@LABELS_JSON@@": _dump(ALL_LABELS),
        "@@ROLE_LABEL_JSON@@": _dump(mbp.ROLE_LABEL),
        "@@SHORT_LABEL_JSON@@": _dump(mbp.SHORT_LABEL),
        "@@ROLE_COLOR_JSON@@": _dump(mbp.ROLE_COLOR),
        "@@XCOLOR_JSON@@": _dump(mbp.XCOLOR),
        "@@THEME_JSON@@": _dump(theme),
        "@@DASHBOARD_JS@@": DASHBOARD_JS,
    }
    for token, value in replacements.items():
        html = html.replace(token, value)
    return html


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    html = build_html()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {os.path.relpath(OUT_PATH, REPO_ROOT)} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
