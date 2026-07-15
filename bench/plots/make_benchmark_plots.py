"""Precision x size benchmark charts for docs/BENCHMARKS.md (+ zh-CN).

Reads ONLY landed raws under bench/results/ (no hand-entered numbers) and writes
deterministic SVGs to docs/assets/plots/. Every figure function is preceded by a
MANIFEST entry naming the exact raw file(s) it consumes; a file listed as
"cross-check only" is read and its value(s) verified/cited in the caption but not
drawn as its own series (documented at the call site, not silently dropped).

If a raw a figure would need does not exist yet, the figure (or that one series)
is skipped and printed as SKIPPED — never interpolated, never fabricated.

Bilingual: every figure is emitted twice, once per LANGS entry. English keeps the
historical filenames (f1_...svg); the zh-CN variant is written alongside it with a
"_zh" suffix (f1_..._zh.svg). All human-visible strings (titles, axis labels,
legend labels, annotations) are looked up from LABELS/ROLE_LABEL/SHORT_LABEL by an
explicit lang key -- technical tokens (model names, fp16/w8a8/GPTQ, "c=64", units)
are shared verbatim between languages. Geometry (figsize/dpi/subplot layout) is
identical between languages; only text content differs.

Usage:
  python bench/plots/make_benchmark_plots.py

Determinism: fixed figsize/dpi, fixed matplotlib svg.hashsalt, no embedded dates,
svg.fonttype=none (text stays text, no embedded font outlines). Run twice and
diff the output directory to confirm byte-identical SVGs. All lookups below are
keyed by explicit strings/lists (never by iterating a dict/set), so the en/zh
parametrization cannot introduce dict-order nondeterminism.
"""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
# Determinism: matplotlib's SVG backend assigns clip-path / marker ids via
# uuid4() unless a fixed hashsalt is set, and embeds today's date unless the
# caller overrides it per-savefig. Both are pinned so re-running this script
# on a different day produces byte-identical output.
matplotlib.rcParams["svg.hashsalt"] = "rwkv-sglang-benchmark-plots"
matplotlib.rcParams["svg.fonttype"] = "none"  # keep <text>, don't embed font outlines
matplotlib.rcParams["font.family"] = "sans-serif"

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory

SVG_METADATA = {"Date": None, "Creator": "rwkv-sglang bench/plots/make_benchmark_plots.py"}
DPI = 100

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO_ROOT, "bench", "results")
OUT_DIR = os.path.join(REPO_ROOT, "docs", "assets", "plots")

# ---------------------------------------------------------------------------
# Languages + fonts
# ---------------------------------------------------------------------------
# Explicit, ordered list (not a set) -- this drives every loop that touches
# output naming, so iteration order is fixed regardless of dict/hash order.
LANGS = ["en", "zh"]

# CJK font handling: svg.fonttype="none" keeps <text> as literal characters, so
# the *browser* picks the rendering font from this CSS font-family list -- it is
# NOT limited to whatever matplotlib resolved locally at generation time. But
# matplotlib DOES need to resolve a real local font to compute correct glyph
# metrics (bbox/kerning) for layout math (legend sizing, tight_layout, text
# anchoring); if it silently fell back to DejaVu Sans (no CJK glyphs) those
# metrics would be wrong even though the SVG text itself renders fine elsewhere.
#
# Tested on this box (fm.findfont, see task notes) rather than assumed: PingFang
# SC is NOT resolvable by matplotlib's font manager here -- it lives in a private
# CoreText-only framework path, not the standard font directories a filesystem
# scan sees -- even though real browsers on this same Mac CAN render it (they use
# CoreText, not a directory scan). Confirmed locally resolvable instead: Hiragino
# Sans GB and Heiti SC (both real system CJK sans fonts). So the stack below
# lists the common cross-platform names FIRST (for browsers on Windows/Linux/other
# Macs that do have them registered as regular fonts), then the two confirmed-
# local fonts (so *this* machine's matplotlib falls through to a real CJK font
# for its own measurements instead of DejaVu Sans), then DejaVu Sans as the last
# resort so Latin/digits/punctuation always render even if every CJK name fails.
FONT_STACK = {
    "en": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "zh": [
        "PingFang SC", "Noto Sans CJK SC", "Noto Sans SC", "Microsoft YaHei",
        "Hiragino Sans GB", "Heiti SC", "DejaVu Sans", "sans-serif",
    ],
}


def _set_lang_fonts(lang):
    matplotlib.rcParams["font.sans-serif"] = FONT_STACK[lang]


def _suffix(lang):
    return "" if lang == "en" else f"_{lang}"


# ---------------------------------------------------------------------------
# Palette — the dataviz skill's validated categorical instance (fixed hue
# order, CVD-safe). One role -> one hue, reused identically across every
# figure so color always means the same precision tier.
# ---------------------------------------------------------------------------
HUE = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "magenta": "#e87ba4",
    "orange": "#eb6834",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
ALERT = "#b3261e"  # distinct from HUE["red"] (=int4_gptq_asym's tier color) --
                    # used only for alert/callout ink (the w4 cliff arrow), never
                    # for a data series, so it can't be mistaken for a legend hue.
NOISE_BAND = "#d8d6cc"  # neutral gray fill for the F5 delta-panel rerun-noise band

ROLE_COLOR = {
    "fp16": HUE["blue"],
    "fp16_state_fp16": HUE["aqua"],
    "int4_rtn": HUE["yellow"],
    "w8a8": HUE["green"],
    "int4_gptq": HUE["violet"],
    "int4_gptq_asym": HUE["red"],
    "hybrid": HUE["magenta"],
    "w4a8_experimental": HUE["orange"],
    # w8g64 first appears in F7/F15 (no earlier figure drew this tier). The 8 HUE
    # slots are all taken by existing roles, so it gets a ninth, visually distinct
    # teal (darker + bluer than both aqua and green; checked for adjacency contrast
    # against both in the F7/F15 layouts where they co-occur).
    "w8g64": "#0b7f8f",
}

# Non-tier chart identities (engines / model families / latency percentiles /
# parallelism configs). These never share a panel with a precision-tier role
# except fp16 (RWKV's own bars stay fp16-blue), so reusing HUE slots here cannot
# create a same-panel ambiguity with the tier that hue means elsewhere.
XCOLOR = {
    "ours": HUE["blue"],        # rwkv-sglang, this repo (same hue as its fp16 tier)
    "theirs": HUE["orange"],    # the competitor series in a 2-engine comparison
    "theirs2": HUE["magenta"],  # third series where one exists (F8's f3_2605 tree)
    "official": INK,            # official reference markers (F8 panel B)
    "qwen": HUE["orange"],      # Qwen3.5 in cross-family charts (F11/F14/F18)
    "qwen_soft": "#f0a97a",     # Qwen3.5 secondary mode (thinking) — lighter tint
    # RWKV's own bf16 "stock path" bar in F11 (its hand kernels don't fire in bf16):
    # a lighter tint of the fp16-blue "ours" identity, so a reader reads "same model
    # family (RWKV, blue), the stock-precision variant" — deliberately parallel to the
    # qwen/qwen_soft pair. Not a precision-tier hue (bf16 has no ROLE_COLOR slot); it
    # only ever co-occurs with its own darker blue + Qwen orange, never a tier legend.
    "ours_soft": "#8bb4e3",
    "p50": HUE["blue"],
    "p99": HUE["orange"],
    "tp2": HUE["green"],
    "pp2": HUE["orange"],
}

# ---------------------------------------------------------------------------
# LABELS — every human-visible string, keyed by an explicit id then by lang.
# Technical tokens (model names, fp16/w8a8/GPTQ, "c=64", tok/s, F-numbers) are
# NOT translated -- both entries hold the same literal string on purpose, so
# there is exactly one place ("LABELS[key]") that owns each piece of copy.
# ---------------------------------------------------------------------------
LABELS = {
    "concurrency": {"en": "concurrency", "zh": "并发数"},
    "tok_s": {"en": "output tok/s", "zh": "输出 tok/s"},
    "peak": {"en": "peak", "zh": "峰值"},
    "faster_hint": {"en": "faster →", "zh": "更快 →"},
    "accurate_hint": {"en": "↑ more accurate", "zh": "↑ 更准确"},
    "ratio_vs_fp16": {"en": "× vs fp16", "zh": "相对 fp16 倍数"},

    # F1
    "f1_suptitle": {
        "en": "Serving concurrency sweep by precision — RTX 5090 (64-in/256-out, wall-clock)",
        "zh": "按精度分面的服务并发扫描 — RTX 5090(64 进/256 出,全程计时)",
    },

    # F2
    "f2_suptitle": {
        "en": "Serving concurrency sweep by precision — RTX 3090 (64-in/256-out, wall-clock)",
        "zh": "按精度分面的服务并发扫描 — RTX 3090(64 进/256 出,全程计时)",
    },
    "f2_panel2_title": {
        "en": "7.2B — the w4 M=64 cliff, and its kernel-level fix (F0055)",
        "zh": "7.2B — w4 M=64 断崖及其内核级修复(F0055)",
    },
    "gptq_cliff_map_label": {
        "en": "int4 GPTQ (w4a16, cliff map)",
        "zh": "int4 GPTQ(w4a16,断崖地图)",
    },
    "cliff_callout": {
        "en": "cliff: c={c1}→{c2}, {v1:,.0f}→{v2:,.0f} tok/s (-{pct:.0f}%)",
        "zh": "断崖:c={c1}→{c2},{v1:,.0f}→{v2:,.0f} tok/s(-{pct:.0f}%)",
    },

    # F3
    "f3_title": {
        "en": "Accuracy vs. speed frontier — RWKV-7 7.2B",
        "zh": "精度-速度前沿 — RWKV-7 7.2B",
    },
    "f3_xlabel": {
        "en": "single-stream output tok/s (bsz1, unless noted)",
        "zh": "单流输出 tok/s(bsz1,另有说明的除外)",
    },
    "math500_ylabel": {"en": "MATH500 avg@64 (%)", "zh": "MATH500 avg@64(%)"},
    "f3_pt_fp16": {"en": "fp16", "zh": "fp16"},
    "f3_pt_state": {"en": "fp16 + STATE_FP16 (W1')", "zh": "fp16 + STATE_FP16(W1')"},
    "f3_pt_gptq": {"en": "int4 GPTQ (symmetric)", "zh": "int4 GPTQ(对称)"},
    "f3_pt_w4a8": {
        "en": "int4 GPTQ + w4a8-tc, capped (experimental)",
        "zh": "int4 GPTQ + w4a8-tc,封顶(实验性)",
    },
    "f3_red_gate": {
        "en": "failed accuracy gate (RED) — default OFF, opt-in",
        "zh": "精度考试未过(RED)——默认关,须手动开启",
    },
    # Caveat BODY only (the card-provenance disclosure) -- the tier name + measured
    # values are prepended in code from the actual plotted point, so each point gets
    # exactly ONE consolidated annotation (name + values + caveat) instead of two
    # separately-positioned text blocks that can collide with each other.
    "f3_note_fp16_state": {
        "en": "speed: 5090 both · accuracy: fp16 from the 2026-07-09 3090 batch\n"
              "(F0055 §0); state_fp16 is 5090 (F0056) — cards differ for fp16 only",
        "zh": "速度:两者皆 5090 · 精度:fp16 来自 2026-07-09 的 3090 批次\n"
              "(F0055 §0);state_fp16 为 5090(F0056)— 仅 fp16 一项跨卡",
    },
    "f3_note_gptq": {
        "en": "speed: 5090 (§4b) · accuracy: 3090 (F0055 §0) — cards differ, shown explicitly",
        "zh": "速度:5090(§4b)· 精度:3090(F0055 §0)— 跨卡,明确标出",
    },
    "f3_note_w4a8": {
        "en": "NOT single-stream: 3090 peak-concurrency (c=128) · F0055 §6",
        "zh": "不是单流:3090 峰值并发(c=128)· F0055 §6",
    },
    "f3_footnote": {
        "en": "int8 w8a8/w8g64 omitted: no MATH500 avg@64 raw exists for 7.2B at this tier\n"
              "(only compression-rate evidence, §2: +0.0041 bpb pooled). See F4 for the 1.5B w8a8 bar.",
        "zh": "略过 int8 w8a8/w8g64:7.2B 在这一档没有落地的 MATH500 avg@64 原始件\n"
              "(只有 §2 的压缩率证据:pooled +0.0041 bpb)。1.5B 的 w8a8 柱见 F4。",
    },

    # F4
    "f4_title": {
        "en": "MATH500 avg@64 by precision and model size",
        "zh": "MATH500 avg@64,按精度与模型尺寸",
    },
    "no_raw_landed": {"en": "no raw\nlanded", "zh": "原始件\n未落地"},
    "fp16_baseline": {"en": "fp16 baseline", "zh": "fp16 基线"},
    "f4_footnote": {
        "en": "1.5B int4-sym/asym (documented as 14.98%/21.99% in §2/F0043) have no landed raw — "
              "omitted, not fabricated.\nMATH500 avg@64 bars mix RTX 5090 and RTX 3090 runs; this project's "
              "own §2 cross-check found the ruler\nagrees within ±0.27pt across cards at matched "
              "config, i.e. card-invariant within this protocol's noise.",
        "zh": "1.5B int4-sym/asym(§2/F0043 中记录为 14.98%/21.99%)没有落地的原始件 — "
              "留空,不回填。\nMATH500 avg@64 的柱子混合了 RTX 5090 与 RTX 3090 的跑数;本项目 §2 "
              "自己的交叉核对发现\n这把尺在匹配配置下跨卡差异在 ±0.27pt 以内,即在这套协议的噪声"
              "范围内卡间不变。",
    },

    # F5
    "f5_title": {
        "en": "Positional compression — 7.2B, fp32 vs fp16 recurrent state",
        "zh": "位置维度压缩曲线 — 7.2B,fp32 对 fp16 循环状态",
    },
    "f5_pt_off": {"en": "fp32 state (default)", "zh": "fp32 状态(默认)"},
    "f5_pt_on": {"en": "fp16 state (RWKV_STATE_FP16)", "zh": "fp16 状态(RWKV_STATE_FP16)"},
    "f5_xlabel": {"en": "token position (bucket midpoint)", "zh": "token 位置(分桶中点)"},
    # plain ASCII "-log2" rather than U+2212/U+2082 -- some CJK sans fonts (e.g. Hiragino
    # Sans GB, verified via fontTools on this box) lack MINUS SIGN/SUBSCRIPT TWO glyphs;
    # ASCII "-"/"2" render identically in meaning and are universally covered.
    "f5_ylabel": {"en": "mean -log2 p (bits/token)", "zh": "平均 -log2 p(bits/token)"},
    "f5_delta_title": {
        "en": "per-bucket Δ (fp16 state - fp32 state)",
        "zh": "逐段 Δ(fp16 状态 - fp32 状态)",
    },
    "f5_delta_ylabel": {"en": "Δ (B-A), bits/token", "zh": "Δ(B-A),bits/token"},
    "f5_noise_band": {"en": "same-flag rerun noise", "zh": "同开关复跑噪声"},

    # F6
    "f6_suptitle": {
        "en": "The 11-GPU fleet — 1.5B fp16 full stack (wall-clock)",
        "zh": "11 卡全覆盖 — 1.5B fp16 手写核全开(全程计时)",
    },
    "f6_panel_single": {"en": "single request (bsz1)", "zh": "单请求(bsz1)"},
    "f6_panel_peak": {"en": "peak over concurrency sweep", "zh": "并发扫描峰值"},

    # F7
    "f7_title": {
        "en": "Single-request speed ladder — 1.5B, RTX 5090 (steady-state)",
        "zh": "单请求速度阶梯 — 1.5B,RTX 5090(稳态)",
    },
    "f7_step_base": {"en": "no fast kernels", "zh": "不开任何加速核"},
    "f7_step_mid": {"en": "+ fused GEMV\n+ sparse FFN", "zh": "+ 融合 GEMV\n+ 稀疏 FFN"},
    "f7_step_lora": {"en": "+ fused LoRA chain", "zh": "+ 融合 LoRA 链"},
    "f7_step_full": {"en": "+ token-shift glue\n+ autotune (full stack)", "zh": "+ token-shift 胶水\n+ 自调优(全开)"},
    "f7_step_w8": {"en": "int8 w8g64\n(prequantized)", "zh": "int8 w8g64\n(预量化)"},
    "f7_step_w4": {"en": "int4\n(prequantized)", "zh": "int4\n(预量化)"},
    "f7_vs_base": {"en": "vs baseline", "zh": "对基线"},

    # F8
    "f8_suptitle": {
        "en": "Albatross (BlinkDL's speed reference) vs rwkv-sglang",
        "zh": "Albatross(BlinkDL 官方速度参照)对 rwkv-sglang",
    },
    "f8a_title": {
        "en": "1.5B single-stream decode per card (Albatross excludes prompt reading, ours includes it — ~3% against us)",
        "zh": "1.5B 单流解码,逐卡(Albatross 列不含读题时间,我们含——约压低我们 3%)",
    },
    "f8a_albatross": {"en": "Albatross", "zh": "Albatross"},
    "f8a_ours": {"en": "rwkv-sglang (full stack)", "zh": "rwkv-sglang(全核栈)"},
    "f8a_t4_note": {
        "en": "Albatross stock kernel won't compile on T4 (sm75) — only ours runs",
        "zh": "Albatross 出厂核在 T4(sm75)编译不过——只有我们能跑",
    },
    "f8a_retuned": {"en": "we re-tuned it for this card", "zh": "我们为这张卡重调过"},
    "f8b_title": {
        "en": "7.2B large-batch grid on a single RTX 5090 — same public code; official headline numbers are RTX Pro 6000 (per Bo Peng)",
        "zh": "7.2B 大批量格,单张 RTX 5090——同一份公开代码;官方招牌数字测于 RTX Pro 6000(据 Bo Peng 本人说明)",
    },
    "f8b_f3a_stock": {"en": "faster3a stock", "zh": "faster3a 出厂"},
    "f8b_f3a_tuned": {"en": "faster3a re-tuned (ours)", "zh": "faster3a 重调(我们)"},
    "f8b_f3_2605": {"en": "faster3_2605 stock", "zh": "faster3_2605 出厂"},
    "f8b_official": {"en": "official chart (RTX Pro 6000, per Bo)", "zh": "官方图表值(RTX Pro 6000,据 Bo)"},
    "f8b_readme": {"en": "README \"{val}+\" (Pro 6000-class claim)", "zh": "README「{val}+」(Pro 6000 级宣称)"},
    "f8b_cls_decode": {"en": "decode", "zh": "解码"},
    "f8b_cls_prefill": {"en": "prefill", "zh": "预填充"},
    "f8b_cls_batch_prefill": {"en": "batch prefill", "zh": "批量预填充"},

    # F9a / F9b
    "f9a_title": {
        "en": "ShareGPT real workload — rwkv-sglang vs vllm-rwkv (1.5B, output tok/s)",
        "zh": "ShareGPT 真实负载 — rwkv-sglang 对 vllm-rwkv(1.5B,输出 tok/s)",
    },
    "f9a_peak": {"en": "peak (all at once)", "zh": "峰值(全部涌入)"},
    "f9a_r16": {"en": "16 req/s", "zh": "16 req/s"},
    "f9a_r16_3090": {"en": "16 req/s (overload on this card)", "zh": "16 req/s(这张卡上已过载)"},
    "f9a_ttft": {"en": "med. TTFT", "zh": "首字延迟中位"},
    "f9a_ttft_note": {
        "en": "white in-bar number = median time to first token (TTFT, ms)",
        "zh": "柱子里的白字 = 首字延迟中位(TTFT,毫秒)",
    },
    "f9b_title": {
        "en": "ShareGPT real workload — fp16 vs int4 GPTQ (output tok/s)",
        "zh": "ShareGPT 真实负载 — fp16 对 int4 GPTQ(输出 tok/s)",
    },
    "f9b_note_01b_rtn": {
        "en": "RTN not measured on ShareGPT (overlay version drift, §4b) — honest gap",
        "zh": "RTN 的 ShareGPT 没有测(overlay 版本漂移,§4b)——如实标注的缺口",
    },

    # F10
    "f10_title": {
        "en": "Multi-GPU scaling — TP / PP (1.5B bf16, 2×L4, 64-in/256-out)",
        "zh": "多卡扩展 — TP / PP(1.5B bf16,2×L4,64 进/256 出)",
    },
    "f10_tp1": {"en": "tp=1 (1 GPU)", "zh": "tp=1(单卡)"},
    "f10_tp2": {"en": "tp=2 (24/24 exact)", "zh": "tp=2(24/24 精确)"},
    "f10_pp2": {"en": "pp=2 (24/24 exact)", "zh": "pp=2(24/24 精确)"},

    # F11
    "f11_suptitle": {
        "en": "RWKV-7 vs Qwen3.5, same engine, RTX 5090 — three readings, kept distinct",
        "zh": "RWKV-7 对 Qwen3.5,同一引擎,RTX 5090——三种读数,分开摆",
    },
    "f11_deploy": {
        "en": "deployment reading — each side's\nfastest available config, bsz1",
        "zh": "部署读数——两边各用\n自己最快的可用配置,bsz1",
    },
    "f11_arch": {
        "en": "architecture reading — same\nprecision (bf16), stock paths, bsz1",
        "zh": "架构读数——同精度(bf16),\n双方原生路径,bsz1",
    },
    "f11_peak": {
        "en": "peak concurrency — same\nprecision (bf16), full sweep",
        "zh": "峰值并发——同精度(bf16),\n完整扫描",
    },
    "f11_rwkv_deploy": {"en": "RWKV-7 fp16 stack + STATE_FP16", "zh": "RWKV-7 fp16 全核栈 + STATE_FP16"},
    "f11_rwkv_bf16": {"en": "RWKV-7 bf16 (stock)", "zh": "RWKV-7 bf16(原生)"},
    "f11_qwen": {"en": "Qwen3.5 bf16 (best available)", "zh": "Qwen3.5 bf16(最佳可用)"},
    "f11_pa_title": {
        "en": "single-stream, bsz1 — deployment vs architecture reading",
        "zh": "单流,bsz1——部署读数 对 架构读数",
    },
    "f11_pb_title": {
        "en": "peak concurrency — bf16, full sweep",
        "zh": "峰值并发——bf16,完整扫描",
    },
    # {p} filled in code from the two plotted bars, never copied from the doc table.
    "f11_v_deploy": {"en": "deployment: RWKV +{p:.0f}%", "zh": "部署读数:RWKV +{p:.0f}%"},
    "f11_v_arch": {"en": "architecture: Qwen3.5 +{p:.0f}%", "zh": "架构读数:Qwen3.5 +{p:.0f}%"},
    "f11_v_peak": {"en": "RWKV +{p:.0f}%", "zh": "RWKV +{p:.0f}%"},
    "f11_at": {"en": "@c{c}", "zh": "@c{c}"},

    # F12
    "f12_suptitle": {
        "en": "Latency under Poisson arrivals — 1.5B, RTX 5090 (512-in/256-out)",
        "zh": "泊松到达下的延迟 — 1.5B,RTX 5090(512 进/256 出)",
    },
    "f12_ttft_title": {"en": "time to first token (log scale)", "zh": "首字延迟(对数轴)"},
    "f12_tpot_title": {"en": "per-token latency", "zh": "每字间隔"},
    "f12_xlabel": {"en": "arrival rate (req/s)", "zh": "到达速率(请求/秒)"},
    "f12_ms": {"en": "ms", "zh": "毫秒"},
    "f12_inf": {"en": "300 at once", "zh": "300 个同时涌入"},
    "f12_p50": {"en": "p50 (typical)", "zh": "p50(典型)"},
    "f12_p99": {"en": "p99 (worst 1%)", "zh": "p99(最差 1%)"},

    # F13
    "f13_title": {
        "en": "Launch-autotune gain per card, per projection shape (kernel-level A/B)",
        "zh": "启动自调优逐卡收益,按投影形状(核级 A/B)",
    },
    "f13_ylabel": {"en": "time saved on shape (%)", "zh": "该形状省下的时间(%)"},
    "f13_zero_note": {
        "en": "honest zeros kept: H100/H200/B200 heuristic already optimal",
        "zh": "如实保留的零:H100/H200/B200 上启发式已是最优",
    },
    "f13_3090_note": {
        "en": "RTX 3090: 0% ± noise (serving-level, 7 runs — no kernel A/B raw)",
        "zh": "RTX 3090:0% ± 噪声(服务级,7 次跑——无核级 A/B 原始件)",
    },
    "f13_shape_legend": {"en": "projection shape:", "zh": "投影形状:"},

    # F14
    "f14_title": {
        "en": "Per-request memory vs context length — formula-derived, not measured",
        "zh": "每请求显存 对 上下文长度——公式推导,非实测",
    },
    "f14_xlabel": {"en": "context length (tokens)", "zh": "上下文长度(token)"},
    "f14_ylabel": {"en": "per-request state / cache (MiB)", "zh": "每请求状态 / 缓存(MiB)"},
    "f14_rwkv15": {"en": "RWKV-7 1.5B state (constant)", "zh": "RWKV-7 1.5B 状态(恒定)"},
    "f14_rwkv72": {"en": "RWKV-7 7.2B state (constant)", "zh": "RWKV-7 7.2B 状态(恒定)"},
    "f14_rwkv_fp16state": {"en": "same, with RWKV_STATE_FP16", "zh": "同上,开 RWKV_STATE_FP16"},
    "f14_qwen": {"en": "Qwen3.5-2B: GDN state + growing KV cache", "zh": "Qwen3.5-2B:GDN 状态 + 增长的 KV cache"},
    "f14_note": {
        "en": "Formula-derived, not measured (an illustration of the equation).\n"
              "RWKV-7 state = L·(2D+64D) elements, constant in context length\n"
              "(1.5B L=24 D=2048; 7.2B L=32 D=4096; §13 / F0056). Qwen3.5-2B\n"
              "(18 GDN + 6 attn layers): KV cache of its 6 attention layers grows\n"
              "linearly with context. §10's measured +4 MiB over 1K→64K corroborates flat.",
        "zh": "公式推导,非实测(是对该公式的图示)。\n"
              "RWKV-7 状态 = L·(2D+64D) 个元素,与上下文长度无关\n"
              "(1.5B L=24 D=2048;7.2B L=32 D=4096;§13 / F0056)。Qwen3.5-2B\n"
              "(18 层 GDN + 6 层 attn):那 6 层 attention 的 KV cache 随上下文\n"
              "线性增长。§10 实测 1K→64K 仅 +4 MiB,佐证其恒定。",
    },
    "f14_grow": {"en": "grows with context →", "zh": "随上下文增长 →"},

    # F15
    "f15_suptitle": {
        "en": "Quantization tiers at a glance — what each tier costs and buys",
        "zh": "量化各档一览——每档付出什么、得到什么",
    },
    "f15a_title": {
        "en": "compression-rate Δ vs same-size fp16 (pooled bpb, lower=better)",
        "zh": "压缩率相对同尺寸 fp16 的增量(池化 bpb,越低越好)",
    },
    "f15a_ylabel": {"en": "Δ pooled bpb vs fp16", "zh": "池化 bpb 相对 fp16 增量"},
    "f15b_title": {
        "en": "1.5B speed vs MATH500 avg@64 (single-stream c=1, RTX 5090)",
        "zh": "1.5B 单流速度 对 MATH500 avg@64(c=1,RTX 5090)",
    },
    "f15b_xlabel": {"en": "single-stream output tok/s (c=1)", "zh": "单流输出 tok/s(c=1)"},
    "f15_vram_note": {
        "en": "marker area ∝ weight footprint (params × byte-width); GB annotated per point",
        "zh": "标记面积与权重占用成正比(参数 × 字节宽);每点已标注 GB",
    },
    "f15b_cross": {
        "en": "int4 point: speed 5090 · accuracy 3090 (ruler card-invariant ±0.27pt, §2)",
        "zh": "int4 那点:速度 5090 · 精度 3090(尺子跨卡不变 ±0.27pt,§2)",
    },
    "f15a_note": {
        "en": "compression barely moves — the metric that hides int4's 1.5B reasoning collapse (see right)",
        "zh": "压缩率几乎不动——正是这个指标掩盖了 int4 在 1.5B 上的推理坍缩(见右)",
    },
    "f15_size_15": {"en": "1.5B", "zh": "1.5B"},
    "f15_size_72": {"en": "7.2B", "zh": "7.2B"},

    # F16
    "f16_suptitle": {
        "en": "int4 GPTQ vs fp16 across the fleet — the crossover, per card (GPTQ ÷ fp16)",
        "zh": "int4 GPTQ 对 fp16 全卡对比——逐卡反超点(GPTQ ÷ fp16)",
    },
    "f16_c1": {"en": "single-stream (c=1)", "zh": "单流(c=1)"},
    "f16_c128": {"en": "high concurrency (c=128)", "zh": "高并发(c=128)"},
    "f16_ylabel": {"en": "int4 GPTQ ÷ fp16 (>1 = int4 faster)", "zh": "int4 GPTQ ÷ fp16(>1 = int4 更快)"},

    # F17
    "f17_title": {
        "en": "vllm-rwkv ÷ rwkv-sglang by concurrency — 1.5B, shared points only",
        "zh": "vllm-rwkv ÷ rwkv-sglang,按并发——1.5B,只画双方共测点",
    },
    "f17_ylabel": {"en": "vllm-rwkv ÷ rwkv-sglang (>1 = vllm-rwkv faster)", "zh": "vllm-rwkv ÷ rwkv-sglang(>1 = vllm-rwkv 更快)"},

    # F18
    "f18_title": {
        "en": "MATH500 avg@64 — RWKV-7 vs Qwen3.5 (protocols differ by design; see caption)",
        "zh": "MATH500 avg@64 — RWKV-7 对 Qwen3.5(两家协议本就不同,见图注)",
    },
    "f18_rwkv_budget": {"en": "RWKV-7 (1,500-tok budget)", "zh": "RWKV-7(1,500 token 预算)"},
    "f18_qwen_direct": {"en": "Qwen3.5 non-thinking (16,384)", "zh": "Qwen3.5 非思考(16,384)"},
    "f18_qwen_thinking": {"en": "Qwen3.5 thinking (16,384, capped floor)", "zh": "Qwen3.5 思考(16,384,截断下限)"},
    "f18_not_measured": {"en": "not measured —\nrun stopped at 25%\n(§13.4)", "zh": "未测出——\n跑到 25% 主动停\n(§13.4)"},
    "f18_trunc": {"en": "truncated", "zh": "截断"},
}


def T(key, lang):
    return LABELS[key][lang]


# Full legend labels, per role.
ROLE_LABEL = {
    "fp16": {"en": "fp16", "zh": "fp16"},
    "fp16_state_fp16": {
        "en": "fp16 + RWKV_STATE_FP16 (W1')", "zh": "fp16 + RWKV_STATE_FP16 (W1')",
    },
    "int4_rtn": {"en": "int4 RTN", "zh": "int4 RTN"},
    "w8g64": {"en": "int8 w8g64 (weight-only)", "zh": "int8 w8g64(仅权重)"},
    "w8a8": {"en": "int8 w8a8", "zh": "int8 w8a8"},
    "int4_gptq": {"en": "int4 GPTQ", "zh": "int4 GPTQ"},
    "int4_gptq_asym": {"en": "int4 GPTQ (asymmetric)", "zh": "int4 GPTQ(非对称)"},
    "hybrid": {"en": "int4 GPTQ (hybrid ffn.v/ffn.k)", "zh": "int4 GPTQ(混合 ffn.v/ffn.k)"},
    "w4a8_experimental": {
        "en": "int4 GPTQ + w4a8-tc (experimental, opt-in)",
        "zh": "int4 GPTQ + w4a8-tc(实验性,opt-in)",
    },
}

# Compact end-of-line tags (Part 2.2 direct labels) -- short by design so they
# fit past the right edge of a 3-panel figure without adding a second legend.
SHORT_LABEL = {
    "fp16": {"en": "fp16", "zh": "fp16"},
    "fp16_state_fp16": {"en": "+STATE_FP16", "zh": "+STATE_FP16"},
    "int4_rtn": {"en": "RTN", "zh": "RTN"},
    "w8a8": {"en": "w8a8", "zh": "w8a8"},
    "int4_gptq": {"en": "GPTQ", "zh": "GPTQ"},
    "w4a8_experimental": {"en": "w4a8-tc", "zh": "w4a8-tc"},
}


def role_label(role, lang):
    return ROLE_LABEL[role][lang]


def short_label(role, lang):
    return SHORT_LABEL[role][lang]


plt.rcParams["axes.edgecolor"] = AXIS
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.color"] = INK_MUTED
plt.rcParams["ytick.color"] = INK_MUTED
plt.rcParams["text.color"] = INK
plt.rcParams["axes.labelcolor"] = INK_SECONDARY


# ---------------------------------------------------------------------------
# Manifest — figure -> exact raw files consumed (relative to bench/results/).
# Printed / existence-checked by main(); kept next to the code that reads it
# so the two can never drift apart silently.
# ---------------------------------------------------------------------------
MANIFEST = {
    "f1_concurrency_5090": [
        "bsz_sweep_0.1b_fp16_5090.json", "bsz_sweep_0.1b_w4gptq_5090.json", "bsz_sweep_0.1b_w4rtn_5090.json",
        "bsz_sweep_fullstack_5090.json", "w1prime_legEf_1.5b_5090.json",
        "bsz_sweep_1.5b_w4gptq_5090.json", "bsz_sweep_1.5b_w4rtn_5090.json", "bsz_sweep_w8a8v2_5090main.json",
        "72b/sweep_72b_fp16_v3_5090.json", "w1prime_legFinal_B_7.2b_5090.json",
        "bsz_sweep_7.2b_w4gptq_5090.json", "bsz_sweep_7.2b_w4gptq_5090_ext.json", "bsz_sweep_7.2b_w4rtn_5090.json",
        "72b/sweep_72b_w8a8_ceil.json", "72b/sweep_72b_w8a8_max.json", "72b/sweep_72b_w8a8.json",
    ],
    "f2_concurrency_3090": [
        "bsz_sweep_1.5b_fp16_3090.json", "bsz_sweep_1.5b_w4gptq_3090.json", "bsz_sweep_1.5b_w4rtn_3090.json",
        "bsz_sweep_7.2b_fp16_3090.json", "bsz_sweep_7.2b_w4rtn_3090.json", "bsz_sweep_7.2b_w4gptq_3090.json",
        "bsz_sweep_7.2b_w4gptq_3090_cliffmap.json", "bsz_sweep_7.2b_w4gptq_3090_cliffmap_fine.json",
        "bsz_sweep_7.2b_w4gptq_3090_cliff_stage1_w4a8.json",
        # consumed for cross-check only (matched control leg for the w4a8 delta), not drawn as
        # its own line -- it reproduces the plotted GPTQ composite within ~1.6% at shared points
        # (1407.0 vs 1429.5 tok/s @c64); see the figure caption.
        "bsz_sweep_7.2b_w4gptq_3090_cliff_stage1_base.json",
    ],
    "f3_accuracy_speed_frontier": [
        "72b/sweep_72b_fp16_v3_5090.json", "math500_avg64_7.2b_fp16.json",
        "w1prime_legFinal_B_7.2b_5090.json", "math500_avg64_7.2b_fp16_stateon.json",
        "bsz_sweep_7.2b_w4gptq_5090.json", "math500_avg64_7.2b_sym.json",
        "bsz_sweep_7.2b_w4gptq_3090_cliff_stage1_w4a8.json", "math500_avg64_7.2b_w4gptq_w4a8capped_3090.json",
    ],
    "f4_math500_ladder": [
        "math500_avg64_5090main.json", "math500_avg64_w8a8_5090main.json",
        "math500_avg64_1.5b_sym.json", "math500_avg64_1.5b_asym.json",
        "math500_avg64_7.2b_fp16.json", "math500_avg64_7.2b_fp16_stateon.json",
        "math500_avg64_7.2b_sym.json", "math500_avg64_7.2b_asym.json", "math500_avg64_7.2b_hybrid_ffnvk.json",
    ],
    "f5_positional_compression_state_precision": [
        "uncheatable_positional_7.2b_fp16_state32_3090.json",
        "uncheatable_positional_7.2b_fp16_state16_3090.json",
    ],
    "f6_fleet": [
        "fleet_main_10cards.json", "bsz_sweep_fullstack_5090.json",
    ],
    "f7_speed_ladder": [
        "ladder_base_5090.log", "ladder_mid_5090.log", "ladder_lora_5090.log",
        "ladder_full_5090.log", "ladder_w8_5090.log", "ladder_w4_5090.log",
    ],
    "f8_albatross": [
        "albatross_fleet_10cards.json", "fleet_main_10cards.json",
        "albatross_3090.md", "albatross_5090/retuned_summary.json",
        "bsz_sweep_fullstack_3090main.json", "bsz_sweep_fullstack_5090.json",
        "albatross_5090/large_batch_grid.json",
    ],
    "f9a_sharegpt_engines": [
        "realload/sglang_5090_inf.json", "realload/sglang_5090_r16.json",
        "realload/vllm_5090_inf.json", "realload/vllm_5090_r16.json",
        "realload/sglang_3090_inf.json", "realload/sglang_3090_r16.json",
        "realload/vllm_3090_inf.json", "realload/vllm_3090_r16.json",
    ],
    "f9b_sharegpt_w4": [
        "sharegpt_0.1b_fp16_5090_rinf.log", "sharegpt_0.1b_fp16_5090_r16.log",
        "sharegpt_0.1b_w4gptq_5090_rinf.log", "sharegpt_0.1b_w4gptq_5090_r16.log",
        "sharegpt_1.5b_fp16_5090_rinf.log", "sharegpt_1.5b_fp16_5090_r16.log",
        "sharegpt_1.5b_w4gptq_5090_rinf.log", "sharegpt_1.5b_w4gptq_5090_r16.log",
        "sharegpt_7.2b_fp16_5090_rinf.log", "sharegpt_7.2b_fp16_5090_r16.log",
        "sharegpt_7.2b_w4gptq_5090_rinf.log", "sharegpt_7.2b_w4gptq_5090_r16.log",
        "sharegpt_1.5b_fp16_3090_rinf.log", "sharegpt_1.5b_fp16_3090_r16.log",
        "sharegpt_1.5b_w4gptq_3090_rinf.log", "sharegpt_1.5b_w4gptq_3090_r16.log",
        "sharegpt_7.2b_fp16_3090_rinf.log", "sharegpt_7.2b_fp16_3090_r16.log",
        "sharegpt_7.2b_w4gptq_3090_rinf.log", "sharegpt_7.2b_w4gptq_3090_r16.log",
    ],
    "f10_tp_pp": ["tppp_l4_main.json"],
    "f11_qwen35_readings": [
        "w1prime_legEf_1.5b_5090.json", "w1prime_legFinal_B_7.2b_5090.json",
        "qwen35/rwkv7_1.5b_bf16_bsz1_5090.json", "qwen35/rwkv7_7.2b_bf16_bsz1_5090.json",
        "qwen35/qwen35_2b_bf16_bsz1_5090.json", "qwen35/qwen35_9b_bf16_bsz1_5090.json",
        "qwen35/rwkv7_1.5b_bf16_sweep_5090.json", "qwen35/qwen35_2b_bf16_sweep_5090.json",
        "qwen35/rwkv7_7.2b_bf16_sweep_5090_v2.json", "qwen35/qwen35_9b_bf16_sweep_5090_v2.json",
    ],
    "f12_latency_poisson": ["pd_mixed_5090.json"],
    "f13_autotune": ["autotune_ab_9cards.json", "autotune_ab_5090.json"],
    # f14 is formula-derived (BENCHMARKS §13's printed state formulas + F0056's
    # doc-recorded measured per-request state constants) — it reads no raw file
    # by design and its caption says so.
    "f14_state_vs_kv": [],
    "f15_quant_tradeoff": [
        "uncheatable_full_fp16_1.5b_5090main.json", "uncheatable_full_w8_1.5b_5090main.json",
        "uncheatable_full_w8a8_1.5b_5090main.json", "uncheatable_full_w4_1.5b_5090main.json",
        "uncheatable_full_fp16_7.2b_5090main.json", "uncheatable_full_w8a8_7.2b_5090main.json",
        "uncheatable_full_w4_7.2b_5090main.json",
        "bsz_sweep_1.5b_fp16_5090.json", "bsz_sweep_w8a8v2_5090main.json", "bsz_sweep_1.5b_w4gptq_5090.json",
        "math500_avg64_5090main.json", "math500_avg64_w8a8_5090main.json", "math500_avg64_1.5b_sym.json",
    ],
    "f16_w4_fleet": [
        "bsz_sweep_1.5b_fp16_3090.json", "bsz_sweep_1.5b_w4gptq_3090.json",
        "bsz_sweep_7.2b_fp16_3090.json", "bsz_sweep_7.2b_w4gptq_3090.json",
        "bsz_sweep_1.5b_fp16_5090.json", "bsz_sweep_1.5b_w4gptq_5090.json",
        "qwen35/rwkv7_7.2b_fp16_fullstack_resweep_5090_v3.json",
        "bsz_sweep_7.2b_w4gptq_5090.json",
        "w4_speed_fleet/T4_1.5b_fp16.json", "w4_speed_fleet/T4_1.5b_w4gptq.json",
        "w4_speed_fleet/L4_1.5b_fp16.json", "w4_speed_fleet/L4_1.5b_w4gptq.json",
        "w4_speed_fleet/A10G_1.5b_fp16.json", "w4_speed_fleet/A10G_1.5b_w4gptq.json",
        "w4_speed_fleet/A100-40GB_1.5b_fp16.json", "w4_speed_fleet/A100-40GB_1.5b_w4gptq.json",
        "w4_speed_fleet/H100_1.5b_fp16.json", "w4_speed_fleet/H100_1.5b_w4gptq_retry.json",
        "w4_speed_fleet/T4_7.2b_fp16_v2.json", "w4_speed_fleet/T4_7.2b_fp16.json",
        "w4_speed_fleet/T4_7.2b_w4gptq_full.json", "w4_speed_fleet/T4_7.2b_w4gptq.json",
        "w4_speed_fleet/L4_7.2b_fp16_v2.json", "w4_speed_fleet/L4_7.2b_fp16_c128.json", "w4_speed_fleet/L4_7.2b_fp16.json",
        "w4_speed_fleet/L4_7.2b_w4gptq_full.json", "w4_speed_fleet/L4_7.2b_w4gptq.json",
        "w4_speed_fleet/A10G_7.2b_fp16_v2.json", "w4_speed_fleet/A10G_7.2b_fp16_c128.json", "w4_speed_fleet/A10G_7.2b_fp16.json",
        "w4_speed_fleet/A10G_7.2b_w4gptq.json",
        "w4_speed_fleet/A100-40GB_7.2b_fp16.json", "w4_speed_fleet/A100-40GB_7.2b_w4gptq.json",
        "w4_speed_fleet/H100_7.2b_fp16.json", "w4_speed_fleet/H100_7.2b_w4gptq.json",
    ],
    "f17_vllmrwkv_ratio": [
        "vllmrwkv/their_sweep_small_5090.json", "vllmrwkv/their_sweep_large_5090.json",
        "vllmrwkv/their_sweep_small_3090_v2.json", "vllmrwkv/their_sweep_large_3090_v2.json",
        "bsz_sweep_fullstack_5090.json", "bsz_sweep_fullstack_3090main.json",
    ],
    "f18_qwen35_math500": [
        "math500_avg64_5090main.json", "math500_avg64_7.2b_fp16.json",
        "qwen35_accuracy/math500_avg64_2b_chatml_direct_5090.json",
        "qwen35_accuracy/math500_avg64_2b_chatml_thinking_5090_v2.json",
        "qwen35_accuracy/math500_avg64_9b_chatml_direct_5090.json",
    ],
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _path(relpath):
    return os.path.join(RESULTS, relpath)


def exists(relpath):
    return os.path.exists(_path(relpath))


def load_rows(relpath):
    """bsz_throughput.py sweep file -> sorted [(concurrency, out_tok_per_s), ...]."""
    with open(_path(relpath)) as f:
        d = json.load(f)
    rows = [(r["concurrency"], r["out_tok_per_s"]) for r in d["rows"]]
    return sorted(rows)


def merge_rows(relpaths_priority):
    """Union concurrency points across files, first file in the list wins on overlap.

    Deterministic: same input list -> same output, regardless of dict/OS iteration
    order (we sort the final result explicitly).
    """
    by_c = {}
    for rp in relpaths_priority:
        for c, v in load_rows(rp):
            if c not in by_c:
                by_c[c] = v
    return sorted(by_c.items())


def math500_pct(relpath):
    """MATH500 avg@64 raw -> (accuracy percent, correct, total)."""
    with open(_path(relpath)) as f:
        d = json.load(f)
    return d["rollout_accuracy"] * 100.0, d["correct_generations"], d["total_generations"]


def sweep_value_at(relpath, concurrency):
    rows = dict(load_rows(relpath))
    return rows[concurrency]


def sweep_peak(rows):
    return max(rows, key=lambda t: t[1])


def load_raw(relpath):
    """Any landed JSON raw, unmodified."""
    with open(_path(relpath)) as f:
        return json.load(f)


# serving_scale.py log table row: " 1024 |    1 |         409.8 | ..." — the
# context=1024 / bsz=1 steady-state decode tok/s (the §3 ladder's own number).
_LADDER_BSZ1_RE = re.compile(r"^\s*1024 \|\s+1 \|\s+([\d.]+)", re.M)


def ladder_bsz1(relpath):
    with open(_path(relpath), errors="replace") as f:
        m = _LADDER_BSZ1_RE.search(f.read())
    return float(m.group(1))


# sglang.bench_serving summary line: "Output token throughput (tok/s):  9560.49"
_SHAREGPT_OUT_RE = re.compile(r"Output token throughput \(tok/s\):\s+([\d.]+)")


def sharegpt_output_toks(relpath):
    with open(_path(relpath), errors="replace") as f:
        m = _SHAREGPT_OUT_RE.search(f.read())
    return float(m.group(1))


def albatross_3090_15b_b1():
    """albatross_3090.md §5 (the internally-consistent all-sizes re-measurement,
    the table the raw itself says feeds comparison.md): 1.5B bsz1 decode tok/s."""
    with open(_path("albatross_3090.md"), errors="replace") as f:
        sec = f.read().split("## 5.", 1)[1]
    m = re.search(r"^\|\s*1\.5B\s*\|\s*1\s*\|\s*([\d.]+)", sec, re.M)
    return float(m.group(1))


def fleet_w4_rows(relpath):
    """w4_speed_fleet/<GPU>_<model>_<cfg>.json -> sorted [(concurrency, tok/s)].
    The file carries a `results` list; every kind=="sweep" entry contributes its
    rows (a follow-up file passed separately handles partial first attempts)."""
    d = load_raw(relpath)
    by_c = {}
    for r in d["results"]:
        if r.get("kind") == "sweep":
            for x in (r.get("rows") or []):
                by_c.setdefault(x["concurrency"], x["out_tok_per_s"])
    return sorted(by_c.items())


def merge_any(relpaths_priority, loader):
    """merge_rows generalized over the row loader: union of concurrency points,
    first file in the list wins on overlap (same first-wins discipline)."""
    by_c = {}
    for rp in relpaths_priority:
        for c, v in loader(rp):
            if c not in by_c:
                by_c[c] = v
    return sorted(by_c.items())


def pooled_bpb(relpath):
    return load_raw(relpath)["overall"]["pooled_bpb"]


# ---------------------------------------------------------------------------
# Shared chart chrome
# ---------------------------------------------------------------------------
_INT_FMT = FuncFormatter(lambda x, pos: f"{x:,.0f}")
CONCURRENCY_TICKS = [1, 32, 64, 128, 256, 384, 512]  # the actual measured points,
                                                       # not a generic power-of-2 comb


def _style_ax(ax, title, lang, logx=True, ylabel_key="tok_s", show_xlabel=True):
    ax.set_title(title, fontsize=13, color=INK, pad=9, loc="left")
    if show_xlabel:
        ax.set_xlabel(T("concurrency", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.set_ylabel(T(ylabel_key, lang), fontsize=11.5, color=INK_SECONDARY)
    if logx:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(CONCURRENCY_TICKS))
        ax.xaxis.set_major_formatter(_INT_FMT)  # plain "64" not "64.0"/"6×10^1"
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        # 256/384/512 sit close together on a log2 scale; a small rotation keeps
        # all 7 curated ticks legible without crowding into each other.
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")
            lbl.set_rotation_mode("anchor")
    ax.yaxis.set_major_formatter(_INT_FMT)
    ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(labelsize=10.5)
    ax.set_facecolor(SURFACE)


def _plot_series(ax, rows, role, lang, linestyle="-", hollow=False, zorder=3, label_override=None,
                  connect=True):
    if not rows:
        return
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    color = ROLE_COLOR[role]
    label = label_override or role_label(role, lang)
    if hollow:
        ax.plot(xs, ys, linestyle=linestyle, linewidth=2, color=color, zorder=zorder,
                 marker="D", markersize=6.5, markerfacecolor=SURFACE, markeredgecolor=color,
                 markeredgewidth=1.6, label=label)
    else:
        multi = len(rows) > 1 and connect
        marker = "*" if not connect else ("o" if len(rows) > 1 else "*")
        msize = 13 if not connect else (7 if len(rows) > 1 else 13)
        ax.plot(xs, ys, linestyle="-" if multi else "None", linewidth=2, color=color,
                 zorder=zorder, marker=marker, markersize=msize, markerfacecolor=color,
                 markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)


def _source_note(fig):
    fig.text(0.995, 0.008, "rwkv-sglang · bench/plots/make_benchmark_plots.py", ha="right",
              va="bottom", fontsize=6.5, color=INK_MUTED, style="italic")


def _dedupe_legend(handles_labels_list, ax, ncol=3, loc="upper center", bbox=(0.5, -0.02),
                    bbox_transform=None):
    seen = {}
    for ax_ in handles_labels_list:
        h, l = ax_.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            seen.setdefault(ll, hh)
    kwargs = dict(loc=loc, bbox_to_anchor=bbox, ncol=ncol, frameon=False, fontsize=10,
                  handlelength=2.2)
    if bbox_transform is not None:
        # figure-fraction anchor (rather than the default axes-fraction) -- once F1/F2
        # panels carry a second, much-shorter ratio sub-panel below them, an
        # axes-fraction offset means something different depending which axes (tall
        # absolute panel vs. narrow ratio panel) it's relative to. Figure-fraction is
        # invariant to that, so the legend Y position is exact regardless of the
        # height_ratios split above it.
        kwargs["bbox_transform"] = bbox_transform
    ax.legend(list(seen.values()), list(seen.keys()), **kwargs)


# ---------------------------------------------------------------------------
# NEW: direct end-labels + peak markers (Part 2.2) and a cliff callout (2.3)
# ---------------------------------------------------------------------------
def _declutter(values, min_gap):
    """Nudge a list of scalar positions apart so sorted neighbors are >= min_gap
    apart, re-centered on the group's original mean. Deterministic: iterates an
    explicit index range and sorts on numeric value only (stable, no ties on
    real float data), so the result is a pure function of `values`.
    """
    n = len(values)
    if n <= 1:
        return list(values)
    order = sorted(range(n), key=lambda i: values[i])
    sorted_vals = [values[i] for i in order]
    for i in range(1, n):
        if sorted_vals[i] - sorted_vals[i - 1] < min_gap:
            sorted_vals[i] = sorted_vals[i - 1] + min_gap
    shift = (sum(values) - sum(sorted_vals)) / n
    sorted_vals = [v + shift for v in sorted_vals]
    out = [0.0] * n
    for rank, i in enumerate(order):
        out[i] = sorted_vals[rank]
    return out


def _end_labels(ax, entries, fontsize=8.8, x_axes=1.018):
    """entries: explicit ordered list of (x, y, color, text) where (x, y) is each
    line's own last data point. Labels are anchored at a FIXED axes-fraction x
    just past the right spine (independent of the data's xlim) via a blended
    transform, so they never depend on how far the data happens to extend --
    only their y position (from the real data value) is decluttered, in points,
    so close-valued lines get readable tags instead of overlapping.
    """
    if not entries:
        return
    fig = ax.figure
    px_per_pt = fig.dpi / 72.0
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    y_pt = [ax.transData.transform((0, y))[1] / px_per_pt for _, y, _, _ in entries]
    min_gap_pt = fontsize * 1.7
    y_pt_adj = _declutter(y_pt, min_gap_pt)
    for (x, y, color, text), y0_pt, y1_pt in zip(entries, y_pt, y_pt_adj):
        ax.annotate(text, xy=(x_axes, y), xycoords=trans, xytext=(3, y1_pt - y0_pt),
                    textcoords="offset points", fontsize=fontsize, color=color, va="center",
                    ha="left", zorder=6, annotation_clip=False, fontweight="bold")


def _mark_peak(ax, rows, color, fmt_val):
    """Small ringed dot + value at a curve's peak -- only when the peak is NOT
    the last point (a peak at the endpoint is already covered by the merged
    end-label+value tag, so a second marker there would just be visual noise).
    """
    if not rows:
        return
    peak = sweep_peak(rows)
    if peak == rows[-1]:
        return
    px, py = peak
    ax.scatter([px], [py], s=50, facecolors=color, edgecolors=SURFACE, linewidths=1.2, zorder=7)
    ax.annotate(fmt_val(py), (px, py), xytext=(0, 9), textcoords="offset points", fontsize=8.4,
                color=color, ha="center", va="bottom", zorder=7, fontweight="bold")


def _extend_xlim_for_labels(ax, factor=1.06):
    """Small log-scale right-margin so the last marker isn't flush against the
    spine. End-labels themselves no longer need this margin (they anchor at a
    fixed axes-fraction x via _end_labels' blended transform), so this is just
    breathing room for the marker/line, not label text -- hence the small factor.
    """
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * factor)


def _ratio_rows(rows, base_rows):
    """`rows` divided by `base_rows` (a panel's fp16 series), point by point --
    ONLY at concurrencies present in both (Phase 2a spec: no interpolation, skip
    non-shared x). `base_rows` and `rows` are both the same sorted [(c, v), ...]
    shape load_rows/merge_rows already return, so this composes directly with
    either without any conversion at the call site.
    """
    base_by_c = dict(base_rows)
    return [(c, v / base_by_c[c]) for c, v in rows if c in base_by_c]


def _style_ratio_ax(ax, lang, ratio_values, show_xlabel=True):
    """Narrow companion sub-panel under an F1/F2 absolute panel: fp16 = 1.0 dashed
    reference; every other tier's ratio curve is plotted by the caller (via
    _plot_series, same role color/marker/linestyle it used in the absolute panel
    above, so a tier's ratio line always reads as a scaled-down twin of its own
    curve up top). `ratio_values` is the flat list of every ratio y-value this
    panel actually plotted -- used only so the y=1.0 reference line is
    guaranteed to stay within the visible range even if a panel's tiers happen
    to sit entirely above or entirely below fp16.
    """
    if show_xlabel:
        ax.set_xlabel(T("concurrency", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.set_ylabel(T("ratio_vs_fp16", lang), fontsize=9.6, color=INK_SECONDARY)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(CONCURRENCY_TICKS))
    ax.xaxis.set_major_formatter(_INT_FMT)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
        lbl.set_rotation_mode("anchor")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2g}×"))
    ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(labelsize=9.2)
    ax.set_facecolor(SURFACE)
    # the fp16 reference itself -- dashed, fp16's own hue, at exactly 1.0 (never a
    # plotted data series: fp16/fp16 is 1.0 at every one of its own points by
    # construction, so a flat reference line says the same thing without the
    # redundant marker noise a real series would add).
    ax.axhline(1.0, color=ROLE_COLOR["fp16"], linestyle="--", linewidth=1.3, alpha=0.8, zorder=2)
    vals = list(ratio_values) + [1.0]
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.18, 0.03)
    ax.set_ylim(lo - pad, hi + pad)


def _cliff_callout(ax, c1, v1, c2, v2, lang):
    """Arrowed callout at a sharp concurrency-cliff edge -- computed straight from
    the two measured points passed in (never a copied/hand-entered number)."""
    pct = (v1 - v2) / v1 * 100.0
    text = T("cliff_callout", lang).format(c1=c1, c2=c2, v1=v1, v2=v2, pct=pct)
    mid_x = (c1 * c2) ** 0.5  # geometric mean -- visually centered on a log-x axis
    mid_y = v1 - (v1 - v2) * 0.28
    ax.annotate(text, xy=(c2, v2), xycoords="data", xytext=(mid_x * 0.62, mid_y * 1.34),
                textcoords="data", fontsize=9, color=ALERT, fontweight="bold", ha="center",
                zorder=9, arrowprops=dict(arrowstyle="-|>", color=ALERT, lw=1.7,
                                           connectionstyle="arc3,rad=-0.18",
                                           shrinkA=2, shrinkB=4))


# ---------------------------------------------------------------------------
# F1 -- per-size concurrency curves, RTX 5090
# ---------------------------------------------------------------------------
def f1_panels():
    """Ordered data for F1's three per-size panels -- the exact role/file
    composition BOTH fig_f1_concurrency_5090 (static SVG) and
    make_interactive.py (dashboard) plot, so the two can never quietly drift
    apart on what "the 1.5B panel" contains. Each panel's `series` is an
    explicit ordered list of (rows, role, connect) with fp16 always listed
    first -- both consumers rely on that position as the ratio-panel/ratio-
    view divisor, so it's asserted, not just assumed, at the one call site
    below that reads it.
    """
    gptq_72b = merge_rows(["bsz_sweep_7.2b_w4gptq_5090.json", "bsz_sweep_7.2b_w4gptq_5090_ext.json"])
    w8a8_72b = merge_rows(["72b/sweep_72b_w8a8_ceil.json", "72b/sweep_72b_w8a8_max.json", "72b/sweep_72b_w8a8.json"])
    panels = [
        {
            "title": "0.1B",  # not translated -- a model-size token, like fp16/w8a8 elsewhere
            "xlim_factor": 1.06,
            "series": [
                (load_rows("bsz_sweep_0.1b_fp16_5090.json"), "fp16", True),
                (load_rows("bsz_sweep_0.1b_w4gptq_5090.json"), "int4_gptq", True),
                (load_rows("bsz_sweep_0.1b_w4rtn_5090.json"), "int4_rtn", True),
            ],
        },
        {
            "title": "1.5B",
            "xlim_factor": 1.06,
            "series": [
                (load_rows("bsz_sweep_fullstack_5090.json"), "fp16", True),
                (load_rows("w1prime_legEf_1.5b_5090.json"), "fp16_state_fp16", False),
                (load_rows("bsz_sweep_1.5b_w4gptq_5090.json"), "int4_gptq", True),
                (load_rows("bsz_sweep_1.5b_w4rtn_5090.json"), "int4_rtn", True),
                (load_rows("bsz_sweep_w8a8v2_5090main.json"), "w8a8", True),
            ],
        },
        {
            "title": "7.2B",
            "xlim_factor": 1.08,
            "series": [
                (load_rows("72b/sweep_72b_fp16_v3_5090.json"), "fp16", True),
                (load_rows("w1prime_legFinal_B_7.2b_5090.json"), "fp16_state_fp16", False),
                (gptq_72b, "int4_gptq", True),
                (load_rows("bsz_sweep_7.2b_w4rtn_5090.json"), "int4_rtn", True),
                (w8a8_72b, "w8a8", True),
            ],
        },
    ]
    for p in panels:
        assert p["series"][0][1] == "fp16", f"{p['title']}: fp16 must be series[0] (ratio divisor)"
    return panels


def fig_f1_concurrency_5090(out_path, lang):
    # 2 rows per column: row 0 is the existing absolute tok/s panel, row 1 is its
    # new (Phase 2a) narrow ratio-vs-fp16 companion, sharing that column's x-axis
    # (view limits) via sharex="col" -- confirmed empirically (see task notes) that
    # this does NOT trigger matplotlib's "not compatible with tight_layout" warning
    # as long as gridspec_kw carries only height_ratios (no explicit hspace/wspace);
    # tight_layout is then free to compute spacing itself, same as the pre-2a figure.
    fig, axes = plt.subplots(2, 3, figsize=(16.4, 7.9), dpi=DPI,
                              gridspec_kw={"height_ratios": [3.0, 1.15]}, sharex="col")

    def tag(role, val):
        return f"{short_label(role, lang)} {val:,.0f}"

    # Two passes. Pass 1 plots data and locks in axis scale/limits/titles for
    # every panel, then the figure gets its suptitle+legend and fig.tight_layout
    # runs ONCE for the whole figure. Only THEN (pass 2) do we compute the
    # end-label declutter math: it reads ax.transData.transform(...) to turn
    # data points into pixel positions, and that transform is only final AFTER
    # tight_layout has resized/repositioned the axes. Doing this in one pass
    # (declutter math right after each panel's own plotting) silently used the
    # PRE-tight_layout transform -- self-consistent at the time, but wrong by
    # render time, so labels for close-valued lines ended up overlapping.
    panel_end_entries = []  # [(ax, [(x,y,color,text), ...]), ...]

    for col, panel in enumerate(f1_panels()):
        series = panel["series"]
        fp16_rows = series[0][0]

        ax = axes[0, col]
        end_entries = []
        for rows, role, connect in series:
            _plot_series(ax, rows, role, lang, connect=connect)
            if connect:
                _mark_peak(ax, rows, ROLE_COLOR[role], lambda v: f"{v:,.0f}")
            if rows:
                x, y = rows[-1]
                end_entries.append((x, y, ROLE_COLOR[role], tag(role, y)))
        # style_ax MUST run before extend_xlim: it sets the log-2 x-scale, and
        # extending/reading xlim while the axis is still linear (the matplotlib
        # default before set_xscale runs) computes nonsense once log scale applies.
        # show_xlabel=False + labelbottom=False: the ratio sub-panel below now owns
        # the "concurrency" axis label and tick labels (same pattern F5 already uses
        # for its ax/axd pair) -- showing them twice would be redundant clutter.
        _style_ax(ax, panel["title"], lang, show_xlabel=False)
        ax.tick_params(labelbottom=False)
        _extend_xlim_for_labels(ax, factor=panel["xlim_factor"])
        panel_end_entries.append((ax, end_entries))

        rax = axes[1, col]
        ratio_vals = []
        for rows, role, connect in series[1:]:  # skip fp16 -- it's the dashed y=1 reference
            rr = _ratio_rows(rows, fp16_rows)
            _plot_series(rax, rr, role, lang, connect=connect)
            ratio_vals.extend(v for _, v in rr)
        _style_ratio_ax(rax, lang, ratio_vals)

    # legend handles come from the top row only (the ratio row reuses identical
    # role colors/labels, so _dedupe_legend's by-label dedup would collapse them
    # anyway -- reading only axes[0, :] just skips that redundant work). Anchored
    # in FIGURE fraction (not axes-fraction) so its position doesn't depend on
    # which row's axes it's nominally attached to.
    _dedupe_legend(axes[0, :], axes[1, 1], ncol=5, loc="upper center", bbox=(0.5, 0.045),
                    bbox_transform=fig.transFigure)
    fig.suptitle(T("f1_suptitle", lang), fontsize=14.5, color=INK, x=0.01, ha="left", y=0.995)
    # right=0.91 (not 1.0): the rightmost panel's end-labels are drawn past its own
    # axes at a fixed axes-fraction x (see _end_labels) with annotation_clip=False,
    # so they are NOT constrained to the axes box -- only to the figure canvas. With
    # no reserved right margin here they got silently cut off by the SVG viewBox for
    # exactly the last panel (the only one with no neighboring panel's white space to
    # bleed into). Confirmed empirically: longest end-label ("+STATE_FP16 7,087")
    # needed ~6.8% of the figure width beyond the last axes; 9% leaves headroom.
    # bottom=0.12 (was 0.09 pre-2a): the ratio row's own rotated tick labels + xlabel
    # now live inside this margin too (the absolute row's copy was removed above),
    # plus the legend anchored at fig-fraction y=0.045 below that.
    fig.tight_layout(rect=(0, 0.12, 0.91, 0.90))
    fig.canvas.draw()  # finalize axes transforms before any pixel-space label math

    for ax_, entries in panel_end_entries:
        _end_labels(ax_, entries)

    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F2 -- per-size concurrency curves, RTX 3090 (incl. the w4 cliff)
# ---------------------------------------------------------------------------
def f2_panels():
    """Ordered data for F2's two per-size panels -- see f1_panels()'s docstring
    for why this is factored out (single source of truth for the static SVG
    and make_interactive.py). Each series tuple is (rows, role, connect,
    hollow, linestyle, label_key); panel 1 (7.2B) is where the non-default
    fields actually get used (the cliff-map GPTQ relabel, the hollow/dashed
    w4a8-tc experimental point). `label_key` is a LABELS key (not a resolved
    string) -- this function is lang-independent, so resolving it to en/zh
    text is left to whichever caller has a `lang` in scope.
    """
    fp16_15b = load_rows("bsz_sweep_1.5b_fp16_3090.json")
    panel0 = {
        "title": "1.5B",
        "series": [
            (fp16_15b, "fp16", True, False, "-", None),
            (load_rows("bsz_sweep_1.5b_w4gptq_3090.json"), "int4_gptq", True, False, "-", None),
            (load_rows("bsz_sweep_1.5b_w4rtn_3090.json"), "int4_rtn", True, False, "-", None),
        ],
    }

    fp16_rows = load_rows("bsz_sweep_7.2b_fp16_3090.json")
    rtn_rows = load_rows("bsz_sweep_7.2b_w4rtn_3090.json")
    gptq_cliff = merge_rows([
        "bsz_sweep_7.2b_w4gptq_3090.json",           # c=1,32 (+128, superseded by cliffmap below)
        "bsz_sweep_7.2b_w4gptq_3090_cliffmap.json",  # c=48,64,80,96,112,128
        "bsz_sweep_7.2b_w4gptq_3090_cliffmap_fine.json",  # c=66,72,76
    ])
    w4a8_exp = load_rows("bsz_sweep_7.2b_w4gptq_3090_cliff_stage1_w4a8.json")
    panel1 = {
        "title_key": "f2_panel2_title",  # unlike panel0's plain "1.5B", this title IS translated
        "series": [
            (fp16_rows, "fp16", True, False, "-", None),
            (rtn_rows, "int4_rtn", True, False, "-", None),
            (gptq_cliff, "int4_gptq", True, False, "-", "gptq_cliff_map_label"),
            (w4a8_exp, "w4a8_experimental", True, True, "--", None),
        ],
        # the cliff edge -- the actual measured c=64 -> c=66 pair (F0055's headline
        # number), read straight off the landed raw, never hand-copied.
        "cliff": {"c1": 64, "v1": dict(gptq_cliff)[64], "c2": 66, "v2": dict(gptq_cliff)[66]},
    }
    for p in (panel0, panel1):
        assert p["series"][0][1] == "fp16", "fp16 must be series[0] (ratio divisor)"
    return [panel0, panel1]


def fig_f2_concurrency_3090(out_path, lang):
    # See fig_f1_concurrency_5090's opening comment for why sharex="col" + a plain
    # height_ratios gridspec_kw (no explicit hspace) coexists safely with the
    # existing tight_layout(rect=...) margin trick used below.
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.3), dpi=DPI,
                              gridspec_kw={"height_ratios": [3.0, 1.15]}, sharex="col")

    def tag(role, val, short=None):
        return f"{short or short_label(role, lang)} {val:,.0f}"

    # Two passes -- see the matching comment in fig_f1_concurrency_5090: end-label
    # declutter math needs the FINAL (post-tight_layout) transform, so it's
    # deferred to a second pass after the whole figure's layout is locked in.
    panel_end_entries = []
    panels = f2_panels()

    # -- 1.5B: fp16 / GPTQ / RTN, matched §4b matrix session, 6pt each --
    ax = axes[0, 0]
    series0 = panels[0]["series"]
    fp16_15b = series0[0][0]
    end_entries = []
    for rows, role, connect, hollow, linestyle, label_key in series0:
        _plot_series(ax, rows, role, lang, connect=connect, hollow=hollow, linestyle=linestyle)
        _mark_peak(ax, rows, ROLE_COLOR[role], lambda v: f"{v:,.0f}")
        if rows:
            x, y = rows[-1]
            end_entries.append((x, y, ROLE_COLOR[role], tag(role, y)))
    _style_ax(ax, panels[0]["title"], lang, show_xlabel=False)
    ax.tick_params(labelbottom=False)
    _extend_xlim_for_labels(ax)
    panel_end_entries.append((ax, end_entries))

    rax = axes[1, 0]
    ratio_vals = []
    for rows, role, connect, hollow, linestyle, label_key in series0[1:]:
        rr = _ratio_rows(rows, fp16_15b)
        _plot_series(rax, rr, role, lang, connect=connect, hollow=hollow, linestyle=linestyle)
        ratio_vals.extend(v for _, v in rr)
    _style_ratio_ax(rax, lang, ratio_vals)

    # -- 7.2B: fp16 / RTN (3pt matrix) + int4-GPTQ cliff composite + w4a8-tc experimental --
    ax = axes[0, 1]
    panel1 = panels[1]
    by_role = {role: (rows, hollow, linestyle, label_key)
               for rows, role, connect, hollow, linestyle, label_key in panel1["series"]}
    fp16_rows = by_role["fp16"][0]
    rtn_rows = by_role["int4_rtn"][0]
    gptq_cliff = by_role["int4_gptq"][0]
    w4a8_exp = by_role["w4a8_experimental"][0]

    _plot_series(ax, fp16_rows, "fp16", lang)
    _plot_series(ax, rtn_rows, "int4_rtn", lang)
    _plot_series(ax, gptq_cliff, "int4_gptq", lang, label_override=T("gptq_cliff_map_label", lang))
    _plot_series(ax, w4a8_exp, "w4a8_experimental", lang, linestyle="--", hollow=True)

    _mark_peak(ax, fp16_rows, ROLE_COLOR["fp16"], lambda v: f"{v:,.0f}")
    _mark_peak(ax, rtn_rows, ROLE_COLOR["int4_rtn"], lambda v: f"{v:,.0f}")
    # gptq_cliff's peak IS the pre-cliff point (c=64) -- that's the headline of this
    # panel, called out explicitly below rather than with a generic peak dot.
    _mark_peak(ax, w4a8_exp, ROLE_COLOR["w4a8_experimental"], lambda v: f"{v:,.0f}")

    _style_ax(ax, T(panel1["title_key"], lang), lang, show_xlabel=False)
    ax.tick_params(labelbottom=False)

    end_entries = []
    for rows, role in ((fp16_rows, "fp16"), (rtn_rows, "int4_rtn")):
        x, y = rows[-1]
        end_entries.append((x, y, ROLE_COLOR[role], tag(role, y)))
    gx, gy = gptq_cliff[-1]
    end_entries.append((gx, gy, ROLE_COLOR["int4_gptq"], tag("int4_gptq", gy, short="GPTQ")))
    wx, wy = w4a8_exp[-1]
    end_entries.append((wx, wy, ROLE_COLOR["w4a8_experimental"], tag("w4a8_experimental", wy)))
    _extend_xlim_for_labels(ax, factor=1.1)
    panel_end_entries.append((ax, end_entries))

    _cliff_callout(ax, panel1["cliff"]["c1"], panel1["cliff"]["v1"],
                    panel1["cliff"]["c2"], panel1["cliff"]["v2"], lang)

    rax = axes[1, 1]
    ratio_vals = []
    for rows, role, connect, hollow, linestyle, label_key in panel1["series"][1:]:
        rr = _ratio_rows(rows, fp16_rows)
        label_override = T(label_key, lang) if label_key else None
        _plot_series(rax, rr, role, lang, linestyle=linestyle, hollow=hollow,
                     label_override=label_override)
        ratio_vals.extend(v for _, v in rr)
    _style_ratio_ax(rax, lang, ratio_vals)

    # y=0.105 (not F1's 0.045): this legend is 3 rows tall (5 unique labels at
    # ncol=2, incl. the cliff-map-relabeled GPTQ entry) vs. F1's single row --
    # confirmed by measuring get_window_extent() post-render (see task notes) that
    # 0.045 put its top edge at ~30px of a 68px-tall legend, clipping ~38px off
    # the bottom of the saved figure; 0.105 gives it full clearance plus margin.
    _dedupe_legend(axes[0, :], axes[1, 1], ncol=2, loc="upper center", bbox=(0.5, 0.105),
                    bbox_transform=fig.transFigure)
    fig.suptitle(T("f2_suptitle", lang), fontsize=14.5, color=INK, x=0.01, ha="left", y=0.995)
    # right=0.885 (was 0.89 pre-2a) -- see the matching comment in
    # fig_f1_concurrency_5090: reserves figure-level margin for the rightmost
    # panel's end-labels (annotation_clip=False means they're bounded by the
    # canvas, not the axes box). Nudged 0.5pt tighter than the pre-2a value:
    # confirmed via get_tightbbox() that the taller (2a) figure pushed the
    # longest end-label ("w4a8-tc 1,468") ~2px past the right canvas edge at
    # 0.89; 0.885 was verified clear with margin to spare. bottom=0.15 (was 0.11
    # pre-2a): the ratio row's ticks/xlabel + the ncol=2 (3-row) legend both now
    # live in this margin, in place of the single absolute row's own copies.
    fig.tight_layout(rect=(0, 0.15, 0.885, 0.90))
    fig.canvas.draw()  # finalize axes transforms before any pixel-space label math

    for ax_, entries in panel_end_entries:
        _end_labels(ax_, entries)

    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F3 -- accuracy-vs-speed frontier, 7.2B
# ---------------------------------------------------------------------------
def f3_points():
    """Ordered points for F3 -- see f1_panels()'s docstring for why this is
    factored out (single source of truth for the static SVG and
    make_interactive.py). Dicts (not the plotting function's own tuple shape)
    so both consumers can read fields by name; `label_key`/`note_key` are
    LABELS keys (lang-independent), resolved via T(key, lang) by whichever
    caller has a `lang` in scope.
    """
    points = []

    x = sweep_value_at("72b/sweep_72b_fp16_v3_5090.json", 1)
    y, c, n = math500_pct("math500_avg64_7.2b_fp16.json")
    points.append({"x": x, "y": y, "role": "fp16", "label_key": "f3_pt_fp16", "hollow": False,
                    "note_key": "f3_note_fp16_state", "correct": c, "total": n})

    x = sweep_value_at("w1prime_legFinal_B_7.2b_5090.json", 1)
    y, c, n = math500_pct("math500_avg64_7.2b_fp16_stateon.json")
    points.append({"x": x, "y": y, "role": "fp16_state_fp16", "label_key": "f3_pt_state", "hollow": False,
                    "note_key": "f3_note_fp16_state", "correct": c, "total": n})

    x = sweep_value_at("bsz_sweep_7.2b_w4gptq_5090.json", 1)
    y, c, n = math500_pct("math500_avg64_7.2b_sym.json")
    points.append({"x": x, "y": y, "role": "int4_gptq", "label_key": "f3_pt_gptq", "hollow": False,
                    "note_key": "f3_note_gptq", "correct": c, "total": n})

    gptq_cliff_w4a8 = load_rows("bsz_sweep_7.2b_w4gptq_3090_cliff_stage1_w4a8.json")
    x = sweep_peak(gptq_cliff_w4a8)[1]
    y, c, n = math500_pct("math500_avg64_7.2b_w4gptq_w4a8capped_3090.json")
    points.append({"x": x, "y": y, "role": "w4a8_experimental", "label_key": "f3_pt_w4a8", "hollow": True,
                    "note_key": "f3_note_w4a8", "correct": c, "total": n, "red_gate": True})
    return points


def fig_f3_accuracy_speed_frontier(out_path, lang):
    fig, ax = plt.subplots(figsize=(9.6, 8.6), dpi=DPI)

    # (x, y, role, label, hollow) -- same shape the rest of this function has
    # always plotted; only the source (f3_points(), shared with the interactive
    # dashboard) and the label resolution (T(...) here, now) are new.
    points = [(p["x"], p["y"], p["role"], T(p["label_key"], lang), p["hollow"]) for p in f3_points()]

    # dashed lineage connector: the experimental w4a8-tc point is int4 GPTQ (symmetric)
    # plus an activation-quant kernel bolted on -- same dashed/hollow convention F2
    # already uses for this role, making the lineage visible instead of implied.
    gptq_pt, w4a8_pt_xy = points[2], points[3]
    ax.plot([gptq_pt[0], w4a8_pt_xy[0]], [gptq_pt[1], w4a8_pt_xy[1]], linestyle="--",
            linewidth=1.3, color=ROLE_COLOR["w4a8_experimental"], alpha=0.55, zorder=2)

    for x, y, role, label, hollow in points:
        color = ROLE_COLOR[role]
        if hollow:
            ax.scatter([x], [y], s=190, facecolors=SURFACE, edgecolors=color, linewidths=2.4,
                       marker="D", zorder=5, label=label)
        else:
            ax.scatter([x], [y], s=160, facecolors=color, edgecolors=SURFACE, linewidths=1.3,
                       marker="o", zorder=5, label=label)

    xmax = max(p[0] for p in points)
    ax.set_xlim(-40, xmax * 1.16)
    ymin = min(p[1] for p in points)
    ymax = max(p[1] for p in points)
    # Asymmetric padding: extra room ABOVE (the fp16/+STATE_FP16 cluster is the
    # topmost point AND carries the longest consolidated label -- name + values +
    # a 2-line card-provenance caveat -- so it needs real clearance from the title).
    span = ymax - ymin
    ax.set_ylim(ymin - (span * 0.55 + 2), ymax + (span * 0.95 + 3))

    fp16_pt, state_pt, gptq_pt, w4a8_pt = points

    # -- one consolidated annotation per point/cluster: tier name + the two
    # measured values (Part 2.4) + the card-provenance caveat, as ONE text block
    # instead of two independently-positioned ones (which collided with each
    # other and with the title in an earlier version of this figure).
    fp16_state_label = (
        f"{T('f3_pt_fp16', lang)} / +STATE_FP16\n"
        f"{fp16_pt[0]:,.0f} tok/s · {fp16_pt[1]:.2f}% / {state_pt[1]:.2f}%\n"
        f"{T('f3_note_fp16_state', lang)}"
    )
    ax.annotate(fp16_state_label, (fp16_pt[0], fp16_pt[1]), textcoords="offset points",
                xytext=(20, 34), fontsize=8.2, color=INK, zorder=6, ha="left", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.7))

    gptq_label = (
        f"{T('f3_pt_gptq', lang)}\n{gptq_pt[0]:,.0f} tok/s · {gptq_pt[1]:.2f}%\n"
        f"{T('f3_note_gptq', lang)}"
    )
    ax.annotate(gptq_label, (gptq_pt[0], gptq_pt[1]), textcoords="offset points", xytext=(18, -40),
                fontsize=8.2, color=INK, zorder=6, ha="left", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.7))

    w4a8_label = (
        f"{T('f3_pt_w4a8', lang)}\n{w4a8_pt[0]:,.0f} tok/s (peak-c) · {w4a8_pt[1]:.2f}%\n"
        f"{T('f3_note_w4a8', lang)}"
    )
    ax.annotate(w4a8_label, (w4a8_pt[0], w4a8_pt[1]), textcoords="offset points", xytext=(-18, -46),
                fontsize=8.2, color=INK, zorder=6, ha="right", linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.7))
    # the RED-gate flag stands alone in ALERT ink, right against the hollow marker --
    # short enough that it doesn't need line-precise stacking under the main label.
    ax.annotate(T("f3_red_gate", lang), (w4a8_pt[0], w4a8_pt[1]), textcoords="offset points",
                xytext=(-18, 14), fontsize=8.6, color=ALERT, zorder=7, ha="right", fontweight="bold")

    ax.set_xlabel(T("f3_xlabel", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.set_ylabel(T("math500_ylabel", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.set_title(T("f3_title", lang), fontsize=13.5, color=INK, loc="left", pad=14)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(labelsize=10.5)
    ax.set_facecolor(SURFACE)

    # faint quadrant guidance, in the margin
    ax.text(0.995, -0.072, T("faster_hint", lang), transform=ax.transAxes, fontsize=9,
            color=INK_MUTED, ha="right", va="top", style="italic")
    ax.text(-0.065, 0.995, T("accurate_hint", lang), transform=ax.transAxes, fontsize=9,
            color=INK_MUTED, ha="left", va="bottom", rotation=90, style="italic")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=10,
              handletextpad=0.5)

    # the legend is 2 rows tall (4 entries, ncol=2) below the axes -- push the
    # footnote well clear of it rather than guessing a tight gap.
    ax.text(0.01, -0.34, T("f3_footnote", lang), transform=ax.transAxes, fontsize=7.4,
            color=INK_MUTED, va="top", ha="left", style="italic")
    fig.tight_layout(rect=(0, 0.17, 1, 0.96))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F4 -- MATH500 accuracy ladder, precision x size (grouped bars)
# ---------------------------------------------------------------------------
def f4_data():
    """Slots + per-size accuracy dicts for F4 -- see f1_panels()'s docstring
    for why this is factored out (single source of truth for the static SVG
    and make_interactive.py). `slot_label` values are technical tokens
    (fp16/w8a8/int4-sym/...) shared verbatim between languages like
    everywhere else in this module, so unlike f1_panels/f2_panels/f3_points
    nothing here needs a separate lang-resolved counterpart.
    """
    slots = ["fp16", "fp16_state_fp16", "w8a8", "int4_gptq", "int4_gptq_asym", "hybrid"]
    slot_label = {
        "fp16": "fp16", "fp16_state_fp16": "+STATE_FP16", "w8a8": "w8a8",
        "int4_gptq": "int4-sym", "int4_gptq_asym": "int4-asym", "hybrid": "hybrid",
    }
    onepfive = {
        "fp16": math500_pct("math500_avg64_5090main.json"),
        "w8a8": math500_pct("math500_avg64_w8a8_5090main.json"),
        # 1.5B int4 raws recovered from the desktop box 2026-07-14 (F0043-era runs; values
        # match BENCHMARKS.md §2/§4 exactly: 4794/32000 and 7036/32000).
        "int4_gptq": math500_pct("math500_avg64_1.5b_sym.json"),
        "int4_gptq_asym": math500_pct("math500_avg64_1.5b_asym.json"),
    }
    seven2b = {
        "fp16": math500_pct("math500_avg64_7.2b_fp16.json"),
        "fp16_state_fp16": math500_pct("math500_avg64_7.2b_fp16_stateon.json"),
        "int4_gptq": math500_pct("math500_avg64_7.2b_sym.json"),
        "int4_gptq_asym": math500_pct("math500_avg64_7.2b_asym.json"),
        "hybrid": math500_pct("math500_avg64_7.2b_hybrid_ffnvk.json"),
    }
    return {"slots": slots, "slot_label": slot_label, "onepfive": onepfive, "seven2b": seven2b}


def fig_f4_math500_ladder(out_path, lang):
    fig, ax = plt.subplots(figsize=(10.6, 7.0), dpi=DPI)

    _d = f4_data()
    slots, slot_label, onepfive, seven2b = _d["slots"], _d["slot_label"], _d["onepfive"], _d["seven2b"]
    missing_1_5b = set()  # 1.5B int4 raws landed 2026-07-14

    n = len(slots)
    bar_w = 0.34
    group_gap = 1.15
    x_1_5b = [i * group_gap for i in range(n)]
    x_7_2b = [i * group_gap + n * group_gap + 1.0 for i in range(n)]

    # fp16 baseline reference line per group (Part 2.5) -- drawn first so bars sit on top
    if "fp16" in onepfive:
        base15, _, _ = onepfive["fp16"]
        ax.hlines(base15, x_1_5b[0] - bar_w * 0.9, x_1_5b[-1] + bar_w * 0.9, color=ROLE_COLOR["fp16"],
                  linestyle="--", linewidth=1.4, alpha=0.4, zorder=2)
    if "fp16" in seven2b:
        base72, _, _ = seven2b["fp16"]
        ax.hlines(base72, x_7_2b[0] - bar_w * 0.9, x_7_2b[-1] + bar_w * 0.9, color=ROLE_COLOR["fp16"],
                  linestyle="--", linewidth=1.4, alpha=0.4, zorder=2)

    for i, slot in enumerate(slots):
        color = ROLE_COLOR[slot]
        if slot in onepfive:
            val, c, tot = onepfive[slot]
            ax.bar(x_1_5b[i], val, width=bar_w, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
            ax.text(x_1_5b[i], val + 1.3, f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
        elif slot in missing_1_5b:
            ax.text(x_1_5b[i], 2.0, T("no_raw_landed", lang), ha="center", va="bottom", fontsize=7.2,
                    color=INK_MUTED, style="italic", rotation=0, linespacing=1.2)

        if slot in seven2b:
            val, c, tot = seven2b[slot]
            ax.bar(x_7_2b[i], val, width=bar_w, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
            ax.text(x_7_2b[i], val + 1.3, f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=INK)

    ax.set_xticks([sum(x_1_5b) / n, sum(x_7_2b) / n])
    ax.set_xticklabels(["1.5B", "7.2B"], fontsize=13, color=INK)
    ax.tick_params(axis="x", length=0, pad=52)
    for i, slot in enumerate(slots):
        ax.text(x_1_5b[i], -3.0, slot_label[slot], ha="right", va="top", fontsize=8.2,
                color=INK_MUTED, rotation=35, rotation_mode="anchor")
        ax.text(x_7_2b[i], -3.0, slot_label[slot], ha="right", va="top", fontsize=8.2,
                color=INK_MUTED, rotation=35, rotation_mode="anchor")

    ax.set_ylabel(T("math500_ylabel", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.set_title(T("f4_title", lang), fontsize=13.5, color=INK, loc="left", pad=15)
    ax.set_ylim(0, 75)
    ax.set_xlim(-0.7, x_7_2b[-1] + 0.7)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=2)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.tick_params(labelsize=10.5)
    ax.set_facecolor(SURFACE)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLOR[s]) for s in slots]
    handles.append(plt.Line2D([0], [0], color=ROLE_COLOR["fp16"], linestyle="--", linewidth=1.4, alpha=0.6))
    legend_labels = [role_label(s, lang) for s in slots] + [T("fp16_baseline", lang)]
    ax.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=3, frameon=False, fontsize=9.6)

    fig.text(0.01, 0.995, T("f4_footnote", lang), fontsize=7.4, color=INK_MUTED, va="top", ha="left",
              style="italic")

    fig.tight_layout(rect=(0, 0.02, 1, 0.84))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F5 (conditional) -- positional compression curve, fp32-state vs fp16-state
# ---------------------------------------------------------------------------
def _bucket_mid(name):
    # "[lo-hi)" -> midpoint; trailing "+" bucket -> its lower edge
    body = name.strip("[)")
    if "-" in body:
        lo, hi = body.rstrip("+").split("-")
        return (int(lo) + int(hi)) / 2
    return float(body.rstrip("+"))


def f5_curve_data():
    """fp32-state vs fp16-state positional curve + delta for F5 -- see
    f1_panels()'s docstring for why this is factored out (single source of
    truth for the static SVG and make_interactive.py). Returns None if the
    two raws haven't landed yet (mirrors fig_f5's own not-yet-landed guard,
    which is why this function -- unlike f1_panels/f2_panels/f3_points/
    f4_data -- can't unconditionally return a value).
    """
    needed = MANIFEST["f5_positional_compression_state_precision"]
    if not all(exists(p) for p in needed):
        return None

    def curve(relpath):
        with open(_path(relpath)) as f:
            return json.load(f)

    off = curve(needed[0])  # fp32 state (leg A)
    on = curve(needed[1])   # fp16 state (leg B)

    rows_by_role = {}
    bucket_ticks = None
    for d, role in ((off, "fp16"), (on, "fp16_state_fp16")):
        buckets = d["position_curve"]
        xs = [_bucket_mid(b["bucket"]) for b in buckets]
        ys = [b["mean_neg_log2_p"] for b in buckets]
        rows_by_role[role] = [(x, y) for x, y in zip(xs, ys) if y == y]  # drop the NaN [3584+) tail bucket
        if bucket_ticks is None:
            bucket_ticks = [x for x, y in zip(xs, ys) if y == y]  # the real bucket midpoints

    # delta: Δ = fp16-state − fp32-state, per bucket, real sample mass only
    on_by_bucket = {b["bucket"]: b for b in on["position_curve"]}
    dxs, dys = [], []
    for b in off["position_curve"]:
        name = b["bucket"]
        a_val = b["mean_neg_log2_p"]
        b_val = on_by_bucket[name]["mean_neg_log2_p"]
        if a_val == a_val and b_val == b_val and b["tokens"] > 0:  # skip the empty NaN tail bucket
            dxs.append(_bucket_mid(name))
            dys.append(b_val - a_val)

    return {
        "bucket_ticks": bucket_ticks,
        "rows_by_role": rows_by_role,  # {"fp16": [(x,y),...], "fp16_state_fp16": [(x,y),...]}
        "delta": {"x": dxs, "y": dys},
        # bits/bucket -- the same-flag rerun band this project measured (F0057 §2:
        # ao3_english reran twice on an unchanged leg, |Δ| <= 6.6e-6, most ~1e-6;
        # ~1e-5 is F0057's own stated conservative headline band).
        "noise": 1e-5,
    }


def fig_f5_positional_compression_state_precision(out_path, lang):
    """7.2B uncheatable position curve, RWKV_STATE_FP16 off (fp32 state) vs on (fp16 state).

    Not yet landed as of this writing -- see MANIFEST key
    f5_positional_compression_state_precision for the two expected filenames. This
    function is intentionally left runnable-but-a-no-op until they land: it checks
    for the raws and returns without writing a file if either is missing, so the
    rest of the batch is never blocked on this one figure.

    Below the main overlapping curves (the "they're identical" claim, by design --
    the two lines are meant to sit on top of each other) a second panel plots the
    per-bucket delta (fp16 state minus fp32 state) so that claim is visible, not
    just asserted: a small-multiples-style secondary axis sharing the x scale,
    computed directly from the two landed raws (never copied from the prose).
    """
    curve_data = f5_curve_data()
    if curve_data is None:
        return False
    bucket_ticks = curve_data["bucket_ticks"]
    rows_by_role = curve_data["rows_by_role"]

    # constrained_layout (not tight_layout) -- this figure stacks two axes via
    # gridspec_kw, and tight_layout's post-hoc rect math is not compatible with a
    # pre-specified gridspec (matplotlib warns and the result can be wrong);
    # constrained_layout is the modern replacement built for exactly this case.
    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(8.4, 7.6), dpi=DPI, sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.35]},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.04, hspace=0.02, wspace=0.0)

    for role, label_key in (("fp16", "f5_pt_off"), ("fp16_state_fp16", "f5_pt_on")):
        _plot_series(ax, rows_by_role[role], role, lang, label_override=T(label_key, lang))
    ax.set_xscale("log", base=2)
    # Without an explicit locator/formatter, matplotlib's default log-2 locator
    # produces crowded/overlapping tick labels at these bucket-midpoint spacings
    # (same class of issue as F1/F2's concurrency axis) -- pin ticks to the actual
    # bucket midpoints instead, same pattern as CONCURRENCY_TICKS.
    ax.xaxis.set_major_locator(FixedLocator(bucket_ticks))
    ax.xaxis.set_major_formatter(_INT_FMT)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(T("f5_ylabel", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.set_title(T("f5_title", lang), fontsize=13.5, color=INK, loc="left", pad=11)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(labelsize=10.5, labelbottom=False)
    ax.set_facecolor(SURFACE)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")

    # -- delta sub-panel: Δ = fp16-state − fp32-state, per bucket, real sample mass only --
    dxs, dys = curve_data["delta"]["x"], curve_data["delta"]["y"]
    noise = curve_data["noise"]
    axd.axhspan(-noise, noise, color=NOISE_BAND, alpha=0.6, zorder=0)
    axd.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    axd.plot(dxs, dys, linestyle="-", linewidth=1.4, color=INK, zorder=3, marker="o",
             markersize=5.5, markerfacecolor=INK, markeredgecolor=SURFACE, markeredgewidth=0.7)

    axd.text(dxs[0], noise, f"  {T('f5_noise_band', lang)}", fontsize=7.6, color=INK_SECONDARY,
              va="bottom", ha="left", style="italic")

    delta_ticks = [-1e-4, -5e-5, 0.0, 5e-5, 1e-4]

    def _delta_fmt(x, pos):
        if x == 0:
            return "0"
        s = f"{x:.0e}"
        return s.replace("e-0", "e-").replace("e+0", "e")

    axd.set_ylim(-1e-4, 1e-4)
    axd.yaxis.set_major_locator(FixedLocator(delta_ticks))
    axd.yaxis.set_major_formatter(FuncFormatter(_delta_fmt))
    # sharex=True syncs the view limits with the top panel but NOT the tick
    # locator/formatter -- axd is the panel that actually draws x labels
    # (labelbottom=False on the top one), so it needs its own explicit pin too.
    axd.xaxis.set_major_locator(FixedLocator(bucket_ticks))
    axd.xaxis.set_major_formatter(_INT_FMT)
    axd.xaxis.set_minor_locator(NullLocator())
    axd.xaxis.set_minor_formatter(NullFormatter())
    for lbl in axd.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
        lbl.set_rotation_mode("anchor")
    axd.set_xlabel(T("f5_xlabel", lang), fontsize=11.5, color=INK_SECONDARY)
    axd.set_ylabel(T("f5_delta_ylabel", lang), fontsize=9.6, color=INK_SECONDARY)
    axd.set_title(T("f5_delta_title", lang), fontsize=9.8, color=INK_SECONDARY, loc="left", pad=6)
    axd.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        axd.spines[spine].set_visible(False)
    axd.spines["left"].set_color(AXIS)
    axd.spines["bottom"].set_color(AXIS)
    axd.tick_params(labelsize=9.5)
    axd.set_facecolor(SURFACE)

    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# F6 -- the 11-GPU fleet (1.5B fp16 full stack): single-request + peak bars
# ---------------------------------------------------------------------------
# Explicit (json_key, display_name) list -- fixed order, never dict iteration.
# Display names match the §6 table's spelling ("RTX PRO 6000", not the JSON's
# dash-separated key).
F6_FLEET_CARDS = [
    ("T4", "T4"), ("L4", "L4"), ("A10G", "A10G"),
    ("A100-40GB", "A100-40GB"), ("A100-80GB", "A100-80GB"), ("L40S", "L40S"),
    ("H100", "H100"), ("H200", "H200"), ("B200", "B200"),
    ("RTX-PRO-6000", "RTX PRO 6000"),
]


def f6_data():
    """11-card fleet rows for F6 -- see f1_panels()'s docstring for why this
    is factored out (single source of truth for the static SVG and
    make_interactive.py). 10 cloud cards from fleet_main_10cards.json (each
    card's runs.sweep_json is the §6 sweep: c=1/32/128/384, 64-in/256-out)
    plus the workstation RTX 5090 from bsz_sweep_fullstack_5090.json (same
    recipe, swept to c=512). Every value is read from those raws; the §6
    table's own cells reproduce from this function.
    """
    fleet = load_raw("fleet_main_10cards.json")
    cards = []
    for key, disp in F6_FLEET_CARDS:
        v = fleet[key]
        rows = sorted((r["concurrency"], r["out_tok_per_s"])
                      for r in v["runs"]["sweep_json"]["rows"])
        peak_c, peak_v = sweep_peak(rows)
        cards.append({"name": disp, "sm": v["sm"], "single": dict(rows)[1],
                      "peak": peak_v, "peak_c": peak_c})
    rows_5090 = load_rows("bsz_sweep_fullstack_5090.json")
    peak_c, peak_v = sweep_peak(rows_5090)
    cards.append({"name": "RTX 5090", "sm": 120, "single": dict(rows_5090)[1],
                  "peak": peak_v, "peak_c": peak_c})
    return cards


def fig_f6_fleet(out_path, lang):
    cards = f6_data()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), dpi=DPI)

    # (field, panel-title key, bar-end tag) -- the peak tag carries the
    # concurrency it was hit at, straight from the raw's own peak row.
    panels = [
        ("single", "f6_panel_single", lambda c: f"{c['single']:,.1f}"),
        ("peak", "f6_panel_peak", lambda c: f"{c['peak']:,.0f} @c{c['peak_c']}"),
    ]
    for ax, (field, title_key, fmt) in zip(axes, panels):
        # each panel sorts independently (ascending -> largest bar on top with
        # barh's bottom-up y): the two panels answer two different ranking
        # questions, so a shared order would misrank one of them.
        order = sorted(cards, key=lambda c: c[field])
        ys = list(range(len(order)))
        vals = [c[field] for c in order]
        # one hue: every bar is the same fp16 tier (identity lives on the y
        # axis, not in color), 2px surface edge = the mark-gap spacer.
        ax.barh(ys, vals, height=0.66, color=ROLE_COLOR["fp16"], zorder=3,
                edgecolor=SURFACE, linewidth=1.0)
        for y, c in zip(ys, order):
            ax.text(c[field], y, f" {fmt(c)}", va="center", ha="left",
                    fontsize=8.6, color=INK, zorder=4)
        ax.set_yticks(ys)
        # GPU name + arch tag in one tick label -- sm is a technical token,
        # shared verbatim between languages like fp16/w8a8 elsewhere.
        ax.set_yticklabels([f"{c['name']}  (sm{c['sm']})" for c in order],
                           fontsize=9.6, color=INK)
        ax.set_title(T(title_key, lang), fontsize=12.5, color=INK, pad=9, loc="left")
        ax.set_xlabel(T("tok_s", lang), fontsize=11.5, color=INK_SECONDARY)
        ax.xaxis.set_major_formatter(_INT_FMT)
        # headroom for the bar-end value tags (the longest is the peak panel's
        # "40,544 @c384" on the topmost bar)
        ax.set_xlim(0, max(vals) * 1.24)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(labelsize=9.6)
        ax.tick_params(axis="y", length=0)
        ax.set_facecolor(SURFACE)

    fig.suptitle(T("f6_suptitle", lang), fontsize=14.5, color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F7 -- the §3 single-request speed ladder, RTX 5090 (ordered bars)
# ---------------------------------------------------------------------------
def f7_data():
    """§3's 5090 ladder column for F7 -- see f1_panels()'s docstring for why
    this is factored out. Each step's tok/s is the context-1024/bsz-1 row of
    its own serving-scale log (ladder_bsz1's regex); the vs-baseline percents
    are recomputed here from those raw values, never copied from the table.
    The table's 3090 column is the v0.5.10 historical ladder with no landed
    per-step raws (repo + boxes checked 2026-07-15), so this figure draws the
    5090 column only -- the doc caption says so.

    Roles: the four stack steps are the fp16 tier; the two prequantized rows
    take their tier hues. The int4 row keeps the doc's own tier-neutral label
    ("int4 (prequantized)") but wears the int4-GPTQ hue readers know from
    F1-F4 -- §4b established GPTQ==RTN speed at every paired point, so the
    hue stands for "the int4 kernel path", not a calibration-method claim.
    """
    steps = [
        ("ladder_base_5090.log", "f7_step_base", "fp16"),
        ("ladder_mid_5090.log", "f7_step_mid", "fp16"),
        ("ladder_lora_5090.log", "f7_step_lora", "fp16"),
        ("ladder_full_5090.log", "f7_step_full", "fp16"),
        ("ladder_w8_5090.log", "f7_step_w8", "w8g64"),
        ("ladder_w4_5090.log", "f7_step_w4", "int4_gptq"),
    ]
    return [{"label_key": key, "role": role, "toks": ladder_bsz1(rp)}
            for rp, key, role in steps]


def fig_f7_speed_ladder(out_path, lang):
    data = f7_data()
    base = data[0]["toks"]
    fig, ax = plt.subplots(figsize=(9.8, 5.6), dpi=DPI)

    n = len(data)
    for i, d in enumerate(data):
        y = n - 1 - i  # top-to-bottom = the table's build order
        ax.barh(y, d["toks"], height=0.62, color=ROLE_COLOR[d["role"]], zorder=3,
                edgecolor=SURFACE, linewidth=1.0)
        if i == 0:
            tag = f" {d['toks']:,.1f}"
        else:
            pct = (d["toks"] / base - 1.0) * 100.0
            tag = f" {d['toks']:,.1f} · +{pct:.1f}% {T('f7_vs_base', lang)}"
        ax.text(d["toks"], y, tag, va="center", ha="left", fontsize=9.2, color=INK, zorder=4)

    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([T(d["label_key"], lang) for d in data], fontsize=9.8, color=INK)
    ax.set_title(T("f7_title", lang), fontsize=13.5, color=INK, pad=11, loc="left")
    ax.set_xlabel(T("tok_s", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.xaxis.set_major_formatter(_INT_FMT)
    ax.set_xlim(0, max(d["toks"] for d in data) * 1.38)  # room for the longest end tag
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_facecolor(SURFACE)

    fig.tight_layout(rect=(0, 0.01, 1, 0.99))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F8 -- Albatross vs rwkv-sglang: §7 per-card single-stream + §7a large-batch
# ---------------------------------------------------------------------------
def f8a_data():
    """§7's per-card single-stream rows -- see f1_panels()'s docstring for why
    this is factored out. Albatross values: albatross_fleet_10cards.json b1
    decode (T4 = compile fail, carried as None); the 3090 from
    albatross_3090.md §5 (the internally-consistent re-measurement, per that
    raw's own note it is the table that feeds comparison.md; §7 flags it as
    re-tuned for the card); the 5090 from retuned_summary.json's stock leg
    (the author's own card -- stock IS its best case there, §7 lead). Ours:
    the same per-card c=1 values F6 draws. Sorted by the Albatross value
    (ascending -> largest on top), T4 pinned last as the no-Albatross row.
    """
    alb = load_raw("albatross_fleet_10cards.json")
    fleet = load_raw("fleet_main_10cards.json")
    rows = []
    for key, disp in F6_FLEET_CARDS:
        a = alb[key]["results"].get("b1", {}).get("decode_tok_s")  # None on T4
        ours = dict(sorted((r["concurrency"], r["out_tok_per_s"])
                           for r in fleet[key]["runs"]["sweep_json"]["rows"]))[1]
        rows.append({"name": disp, "sm": fleet[key]["sm"], "alb": a, "ours": ours})
    rows.append({"name": "RTX 3090", "sm": 86, "alb": albatross_3090_15b_b1(),
                 "ours": dict(load_rows("bsz_sweep_fullstack_3090main.json"))[1],
                 "alb_retuned": True})
    rows.append({"name": "RTX 5090", "sm": 120,
                 "alb": load_raw("albatross_5090/retuned_summary.json")["1.5b/b1/decode"]["stock_tok_s"],
                 "ours": dict(load_rows("bsz_sweep_fullstack_5090.json"))[1]})
    with_alb = sorted((r for r in rows if r["alb"] is not None), key=lambda r: r["alb"])
    no_alb = [r for r in rows if r["alb"] is None]
    return no_alb + with_alb  # bottom-up barh order: T4 (no bar) at the bottom


# §7a's shape rows -- explicit order (the table's own), never dict iteration.
# (grid_case, display, class_label_key, readme_key)
F8B_SHAPES = [
    ("B1T1", "1×1", "f8b_cls_decode", None),
    ("B8T1", "8×1", "f8b_cls_decode", None),
    ("B32T1", "32×1", "f8b_cls_decode", None),
    ("B64T1", "64×1", "f8b_cls_decode", None),
    ("B128T1", "128×1", "f8b_cls_decode", None),
    ("B256T1", "256×1", "f8b_cls_decode", None),
    ("B1024T1", "1024×1", "f8b_cls_decode", "B1024T1_decode"),
    ("B1T256", "1×256", "f8b_cls_prefill", None),
    ("B1T1024", "1×1024", "f8b_cls_prefill", "B1T1024_prefill"),
    ("B16T16", "16×16", "f8b_cls_batch_prefill", None),
    ("B32T32", "32×32", "f8b_cls_batch_prefill", "B32T32_batch_prefill"),
]


def f8b_data():
    """§7a's large-batch grid -- see f1_panels()'s docstring for why this is
    factored out. Each variant's value per shape = mean of the two process
    repeats (the table's own protocol; repeats agree within 0.82% worst
    case). faster3a's B1024 cells come from their dedicated fresh-process
    run groups (the disclosed OOM note in §7a). Official Pro 6000 chart
    values + README "N+" claims are carried verbatim from the raw's own
    reference_numbers block -- reference markers, not our measurements.
    """
    d = load_raw("albatross_5090/large_batch_grid.json")
    grid = d["grid"]

    def mean_case(group, case):
        vals = [grid[group][run]["cases"][case]["tok_s_p50"] for run in ("run1", "run2")]
        return sum(vals) / len(vals)

    def series(main_group, b1024_group):
        out = {}
        for case, _disp, _cls, _readme in F8B_SHAPES:
            group = b1024_group if (case == "B1024T1" and b1024_group) else main_group
            out[case] = mean_case(group, case)
        return out

    ref = d["reference_numbers"]
    official = ref["bo_zhihu_pro6000_2026-07-10"]  # {"B1T1": 144.04, ...}
    readme = ref["albatross_readme_5090_claims"]   # {"B1024T1_decode": "15000+", ...}
    return {
        "f3a_stock": series("f3a_stock_72b", "f3a_stock_1024"),
        "f3a_tuned": series("f3a_tuned_72b", "f3a_tuned_1024"),
        "f3_2605": series("f3_2605_stock_72b", None),
        "official": {k: v for k, v in official.items() if k != "note"},
        "readme": {k: v for k, v in readme.items() if k != "note"},
    }


def fig_f8_albatross(out_path, lang):
    rows = f8a_data()
    b = f8b_data()
    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(13.2, 12.6), dpi=DPI,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )

    # ---- panel A: per-card single-stream, grouped horizontal bars ----
    n = len(rows)
    ys = list(range(n))
    bar_h = 0.34
    for y, r in zip(ys, rows):
        if r["alb"] is not None:
            axa.barh(y + 0.19, r["alb"], height=bar_h, color=XCOLOR["theirs"], zorder=3,
                     edgecolor=SURFACE, linewidth=0.8,
                     label=T("f8a_albatross", lang) if y == ys[-1] else None)
            alb_tag = f" {r['alb']:,.1f}"
            if r.get("alb_retuned"):
                # §7's own table note, carried into the figure: this cell is the
                # value AFTER our per-card re-tune, not out-of-box.
                alb_tag += f" ({T('f8a_retuned', lang)})"
            axa.text(r["alb"], y + 0.19, alb_tag, va="center", ha="left",
                     fontsize=7.8, color=INK, zorder=4)
        else:
            axa.text(r["ours"] + 14, y + 0.19, T("f8a_t4_note", lang), va="center", ha="left",
                     fontsize=7.8, color=INK_SECONDARY, style="italic", zorder=4)
        axa.barh(y - 0.19, r["ours"], height=bar_h, color=XCOLOR["ours"], zorder=3,
                 edgecolor=SURFACE, linewidth=0.8,
                 label=T("f8a_ours", lang) if y == ys[-1] else None)
        axa.text(r["ours"], y - 0.19, f" {r['ours']:,.1f}", va="center", ha="left",
                 fontsize=7.8, color=INK, zorder=4)
        if r["alb"] is not None:
            # ours ÷ Albatross, computed from the two plotted values -- right column
            axa.text(0.995, y, f"{r['ours'] / r['alb']:.2f}×", transform=blended_transform_factory(
                axa.transAxes, axa.transData), va="center", ha="right", fontsize=8.6,
                color=INK_SECONDARY, fontweight="bold", zorder=4)
    axa.set_yticks(ys)
    axa.set_yticklabels([f"{r['name']}  (sm{r['sm']})" for r in rows], fontsize=9.4, color=INK)
    axa.set_title(T("f8a_title", lang), fontsize=11.6, color=INK, pad=9, loc="left")
    axa.set_xlabel(T("tok_s", lang), fontsize=11, color=INK_SECONDARY)
    axa.xaxis.set_major_formatter(_INT_FMT)
    axa.set_xlim(0, max(r["alb"] or 0 for r in rows) * 1.22)
    axa.set_ylim(-0.62, n - 0.38)
    axa.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        axa.spines[spine].set_visible(False)
    axa.spines["left"].set_color(AXIS)
    axa.spines["bottom"].set_color(AXIS)
    axa.tick_params(labelsize=9.4)
    axa.tick_params(axis="y", length=0)
    axa.set_facecolor(SURFACE)
    axa.legend(loc="lower right", frameon=False, fontsize=9.6, bbox_to_anchor=(0.995, 0.02))

    # ---- panel B: §7a large-batch grid, dot plot on a log axis ----
    # (position encodes value honestly across the 84 -> 21,000+ span; bar
    # lengths on a log axis would not)
    yb = list(range(len(F8B_SHAPES)))[::-1]  # table order top -> bottom
    for y, (case, disp, cls_key, readme_key) in zip(yb, F8B_SHAPES):
        axb.axhline(y, color=GRID, linewidth=0.7, zorder=0)
        axb.scatter([b["f3a_stock"][case]], [y], s=64, color=XCOLOR["theirs"], zorder=4,
                    edgecolors=SURFACE, linewidths=0.9)
        axb.scatter([b["f3a_tuned"][case]], [y], s=74, facecolors=SURFACE, zorder=5,
                    edgecolors=XCOLOR["theirs"], linewidths=1.7)
        axb.scatter([b["f3_2605"][case]], [y], s=64, color=XCOLOR["theirs2"], zorder=3,
                    edgecolors=SURFACE, linewidths=0.9)
        if case in b["official"]:
            v = b["official"][case]
            axb.scatter([v], [y], s=86, marker="D", color=XCOLOR["official"], zorder=6,
                        edgecolors=SURFACE, linewidths=0.9)
            axb.annotate(f"{v:,.0f}", (v, y), xytext=(0, 8), textcoords="offset points",
                         fontsize=8, color=INK, ha="center", fontweight="bold", zorder=6)
        if readme_key:
            claim = b["readme"][readme_key]          # e.g. "15000+"
            v = float(claim.rstrip("+"))
            axb.scatter([v], [y], s=92, marker=">", facecolors=SURFACE, zorder=6,
                        edgecolors=XCOLOR["official"], linewidths=1.5)
            axb.annotate(T("f8b_readme", lang).format(val=f"{v:,.0f}"), (v, y), xytext=(0, 9),
                         textcoords="offset points", fontsize=7.4, color=INK_SECONDARY,
                         ha="center", zorder=6)
    axb.set_yticks(yb)
    axb.set_yticklabels([f"{disp}  ·  {T(cls_key, lang)}" for _c, disp, cls_key, _r in F8B_SHAPES],
                        fontsize=9.4, color=INK)
    axb.set_xscale("log")
    xticks = [100, 300, 1000, 3000, 10000, 20000]
    axb.xaxis.set_major_locator(FixedLocator(xticks))
    axb.xaxis.set_major_formatter(_INT_FMT)
    axb.xaxis.set_minor_locator(NullLocator())
    axb.xaxis.set_minor_formatter(NullFormatter())
    axb.set_xlim(70, 30000)
    axb.set_ylim(-0.7, len(F8B_SHAPES) - 0.3)
    axb.set_title(T("f8b_title", lang), fontsize=11.6, color=INK, pad=9, loc="left")
    axb.set_xlabel(T("tok_s", lang) + " (log)", fontsize=11, color=INK_SECONDARY)
    axb.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        axb.spines[spine].set_visible(False)
    axb.spines["left"].set_color(AXIS)
    axb.spines["bottom"].set_color(AXIS)
    axb.tick_params(labelsize=9.4)
    axb.tick_params(axis="y", length=0)
    axb.set_facecolor(SURFACE)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                   markerfacecolor=XCOLOR["theirs"], markeredgecolor=SURFACE,
                   label=T("f8b_f3a_stock", lang)),
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=8.6,
                   markerfacecolor=SURFACE, markeredgecolor=XCOLOR["theirs"], markeredgewidth=1.7,
                   label=T("f8b_f3a_tuned", lang)),
        plt.Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                   markerfacecolor=XCOLOR["theirs2"], markeredgecolor=SURFACE,
                   label=T("f8b_f3_2605", lang)),
        plt.Line2D([0], [0], marker="D", linestyle="None", markersize=8.4,
                   markerfacecolor=XCOLOR["official"], markeredgecolor=SURFACE,
                   label=T("f8b_official", lang)),
    ]
    axb.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.115),
               ncol=2, frameon=False, fontsize=9.2)

    fig.suptitle(T("f8_suptitle", lang), fontsize=14.5, color=INK, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F9a -- ShareGPT real workload: rwkv-sglang vs vllm-rwkv (§7c)
# F9b -- ShareGPT real workload: fp16 vs int4 GPTQ (§4b)
# ---------------------------------------------------------------------------
def f9a_data():
    """§7c's engine comparison -- see f1_panels()'s docstring for why this is
    factored out. Reads the eight realload JSONs (sglang.bench_serving's own
    output): output tok/s + median TTFT per (card, load, engine)."""
    cards = []
    for card in ["5090", "3090"]:
        loads = []
        for rate in ["inf", "r16"]:
            pair = {}
            for eng in ["sglang", "vllm"]:
                d = load_raw(f"realload/{eng}_{card}_{rate}.json")
                pair[eng] = {"out": d["output_throughput"], "ttft_med": d["median_ttft_ms"]}
            loads.append({"rate": rate, "engines": pair})
        cards.append({"card": f"RTX {card}", "loads": loads})
    return cards


def fig_f9a_sharegpt_engines(out_path, lang):
    data = f9a_data()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.5), dpi=DPI)

    def ttft_tag(ms):
        # ms precision follows the value's own scale (31.6 ms vs 2,503 ms),
        # matching how §7c's table prints them
        val = f"{ms:,.1f}" if ms < 100 else f"{ms:,.0f}"
        return f"{val} ms"

    for ax, card in zip(axes, data):
        is_3090 = card["card"] == "RTX 3090"
        xs = [0.0, 1.0]
        ymax_panel = max(e["out"] for l in card["loads"] for e in l["engines"].values())
        for xi, load in zip(xs, card["loads"]):
            for dx, eng, color in ((-0.19, "sglang", XCOLOR["ours"]), (0.19, "vllm", XCOLOR["theirs"])):
                e = load["engines"][eng]
                label = ("rwkv-sglang" if eng == "sglang" else "vllm-rwkv") if xi == 0 else None
                ax.bar(xi + dx, e["out"], width=0.34, color=color, zorder=3,
                       edgecolor=SURFACE, linewidth=1.0, label=label)
                ax.text(xi + dx, e["out"], f"{e['out']:,.0f}", ha="center", va="bottom",
                        fontsize=8.2, color=INK, zorder=4)
                # median TTFT lives INSIDE its own bar (white, near the base) --
                # collision-proof where above-bar sub-labels of near-equal bars
                # were not; the footnote below the figure decodes it.
                ax.text(xi + dx, ymax_panel * 0.025, ttft_tag(e["ttft_med"]), ha="center",
                        va="bottom", fontsize=7.4, color=SURFACE, zorder=4, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([
            T("f9a_peak", lang),
            T("f9a_r16_3090" if is_3090 else "f9a_r16", lang),
        ], fontsize=9.8, color=INK)
        ax.set_title(card["card"], fontsize=12.5, color=INK, pad=9, loc="left")
        ax.set_ylabel(T("tok_s", lang), fontsize=11, color=INK_SECONDARY)
        ax.yaxis.set_major_formatter(_INT_FMT)
        ax.set_ylim(0, max(e["out"] for l in card["loads"] for e in l["engines"].values()) * 1.26)
        ax.set_xlim(-0.62, 1.62)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(labelsize=9.6)
        ax.tick_params(axis="x", length=0)
        ax.set_facecolor(SURFACE)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9.6)

    fig.text(0.01, 0.012, T("f9a_ttft_note", lang), fontsize=7.4, color=INK_MUTED,
             style="italic", ha="left", va="bottom")
    fig.suptitle(T("f9a_title", lang), fontsize=13.5, color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# §4b ShareGPT panels: (card, display model, filename slug). 0.1B exists on
# the 5090 only; the 3090 ran 1.5B/7.2B (that box's §4b scope).
F9B_PANELS = [
    ("5090", "0.1B", "0.1b"), ("5090", "1.5B", "1.5b"), ("5090", "7.2B", "7.2b"),
    ("3090", "1.5B", "1.5b"), ("3090", "7.2B", "7.2b"),
]


def f9b_data():
    """§4b's ShareGPT fp16-vs-GPTQ matrix -- see f1_panels()'s docstring for
    why this is factored out. Each value is the 'Output token throughput'
    line of its own bench_serving log; the Δ% annotations are recomputed
    from the two plotted bars at draw time."""
    out = []
    for card, disp, slug in F9B_PANELS:
        entry = {"card": card, "model": disp, "vals": {}}
        for role, cfg in (("fp16", "fp16"), ("int4_gptq", "w4gptq")):
            for rate in ("rinf", "r16"):
                entry["vals"][(role, rate)] = sharegpt_output_toks(
                    f"sharegpt_{slug}_{cfg}_{card}_{rate}.log")
        out.append(entry)
    return out


def fig_f9b_sharegpt_w4(out_path, lang):
    data = f9b_data()
    fig, axes = plt.subplots(1, 5, figsize=(14.2, 4.9), dpi=DPI)

    for ax, entry in zip(axes, data):
        vals = entry["vals"]
        xs = [0.0, 1.0]
        ymax = max(vals.values())
        for xi, rate in zip(xs, ("rinf", "r16")):
            fp = vals[("fp16", rate)]
            w4 = vals[("int4_gptq", rate)]
            ax.bar(xi - 0.19, fp, width=0.34, color=ROLE_COLOR["fp16"], zorder=3,
                   edgecolor=SURFACE, linewidth=1.0)
            ax.bar(xi + 0.19, w4, width=0.34, color=ROLE_COLOR["int4_gptq"], zorder=3,
                   edgecolor=SURFACE, linewidth=1.0)
            ax.text(xi - 0.19, fp, f"{fp:,.0f}", ha="center", va="bottom", fontsize=7.3,
                    color=INK, zorder=4)
            # GPTQ ÷ fp16 delta, computed from the two plotted bars
            pct = (w4 / fp - 1.0) * 100.0
            ax.text(xi + 0.19, w4, f"{w4:,.0f}\n{pct:+.1f}%", ha="center", va="bottom",
                    fontsize=7.3, color=INK, zorder=4, linespacing=1.3)
        ax.set_xticks(xs)
        ax.set_xticklabels([T("f9a_peak", lang), T("f9a_r16", lang)], fontsize=8.6, color=INK)
        ax.set_title(f"{entry['card']} · {entry['model']}", fontsize=11.5, color=INK,
                     pad=8, loc="left")
        if ax is axes[0]:
            ax.set_ylabel(T("tok_s", lang), fontsize=11, color=INK_SECONDARY)
        ax.yaxis.set_major_formatter(_INT_FMT)
        ax.set_ylim(0, ymax * 1.3)
        ax.set_xlim(-0.62, 1.62)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(labelsize=8.4)
        ax.tick_params(axis="x", length=0)
        ax.set_facecolor(SURFACE)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLOR["fp16"]),
               plt.Rectangle((0, 0), 1, 1, color=ROLE_COLOR["int4_gptq"])]
    fig.legend(handles, [role_label("fp16", lang), role_label("int4_gptq", lang)],
               loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False,
               fontsize=9.6)
    # the honest RTN gap (§4b: overlay version drift blocked the 0.1B RTN pair)
    fig.text(0.01, 0.012, T("f9b_note_01b_rtn", lang), fontsize=7.2, color=INK_MUTED,
             style="italic", ha="left", va="bottom")

    fig.suptitle(T("f9b_title", lang), fontsize=13.5, color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.075, 1, 0.91))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F10 -- multi-GPU TP / PP scaling (§6b)
# ---------------------------------------------------------------------------
def f10_data():
    """§6b's TP/PP matrix -- see f1_panels()'s docstring for why this is
    factored out. tppp_l4_main.json carries, per config, the c1/8/32/64
    throughputs plus the greedy-vs-single-GPU verdict; the vs-tp=1 ratio at
    c=64 is recomputed from the plotted values."""
    d = load_raw("tppp_l4_main.json")
    series = []
    for key, label_key, role in (("tp1", "f10_tp1", "ours"),
                                 ("tp2", "f10_tp2", "tp2"),
                                 ("pp2", "f10_pp2", "pp2")):
        thr = d[key]["throughput"]  # {"1": v, "8": v, "32": v, "64": v}
        rows = [(int(c), thr[c]) for c in ("1", "8", "32", "64")]
        series.append({"key": key, "label_key": label_key, "color": XCOLOR[role],
                       "rows": rows, "matches": d[key].get("matches_tp1")})
    return series


def fig_f10_tp_pp(out_path, lang):
    series = f10_data()
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=DPI)

    base_c64 = dict(series[0]["rows"])[64]
    end_entries = []
    for s in series:
        xs = [r[0] for r in s["rows"]]
        ys = [r[1] for r in s["rows"]]
        ax.plot(xs, ys, linestyle="-", linewidth=2, color=s["color"], zorder=3,
                marker="o", markersize=7, markerfacecolor=s["color"],
                markeredgecolor=SURFACE, markeredgewidth=0.8,
                label=T(s["label_key"], lang))
        ratio = ys[-1] / base_c64
        tag = f"{ys[-1]:,.0f}" if s["key"] == "tp1" else f"{ys[-1]:,.0f} · {ratio:.2f}×"
        end_entries.append((xs[-1], ys[-1], s["color"], tag))

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([1, 8, 32, 64]))
    ax.xaxis.set_major_formatter(_INT_FMT)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(_INT_FMT)
    ax.set_xlabel(T("concurrency", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.set_ylabel(T("tok_s", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.set_title(T("f10_title", lang), fontsize=13, color=INK, pad=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(labelsize=10.5)
    ax.set_facecolor(SURFACE)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    _extend_xlim_for_labels(ax, factor=1.09)

    # right=0.88: the zh end labels ("3,026 · 1.17×") measure ~11px wider than
    # the en ones under the CJK font stack -- verified via the overflow gate,
    # 0.90 clipped them on the zh variant only.
    fig.tight_layout(rect=(0, 0.01, 0.88, 0.99))
    fig.canvas.draw()  # finalize transforms before pixel-space label math
    _end_labels(ax, end_entries)
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F12 -- latency under Poisson arrivals (§9)
# ---------------------------------------------------------------------------
def f12_data():
    """§9's Poisson table -- see f1_panels()'s docstring for why this is
    factored out. pd_mixed_5090.json's rows carry one extra measured rate
    (4 req/s) the table's prose omits; the figure draws every landed row."""
    d = load_raw("pd_mixed_5090.json")
    return d["rows"]  # [{rate, out_tok_per_s, ttft_p50_ms, ttft_p99_ms, tpot_p50_ms, tpot_p99_ms}]


def fig_f12_latency_poisson(out_path, lang):
    rows = f12_data()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0), dpi=DPI,
                              gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})

    xs = list(range(len(rows)))
    xticklabels = [T("f12_inf", lang) if r["rate"] == "inf" else f"{r['rate']:g}" for r in rows]

    def latency_panel(ax, p50_key, p99_key, title_key, logy):
        for key, xkey, color in ((p50_key, "f12_p50", XCOLOR["p50"]), (p99_key, "f12_p99", XCOLOR["p99"])):
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, linestyle="-", linewidth=2, color=color, zorder=3,
                    marker="o", markersize=6.5, markerfacecolor=color,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=T(xkey, lang))
            for x, y in zip(xs, ys):
                # p99 tags above their points, p50 tags below -- fixed offsets,
                # the two series never cross in this data. The final (all-at-
                # once) point sits atop a near-vertical segment, so its tag
                # goes to the LEFT of the marker instead.
                val = f"{y:,.0f}" if y >= 100 else f"{y:g}"
                if x == xs[-1]:
                    ax.annotate(val, (x, y), xytext=(-7, 0), textcoords="offset points",
                                fontsize=7.4, color=color, ha="right", va="center", zorder=5)
                else:
                    dy = 8 if key == p99_key else -14
                    ax.annotate(val, (x, y), xytext=(0, dy), textcoords="offset points",
                                fontsize=7.4, color=color, ha="center", zorder=5)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([10, 30, 100, 300, 1000, 3000]))
            ax.yaxis.set_major_formatter(_INT_FMT)
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_major_formatter(_INT_FMT)
            ax.set_ylim(0, max(r[p99_key] for r in rows) * 1.28)
        ax.set_title(T(title_key, lang), fontsize=11.6, color=INK, pad=9, loc="left")
        ax.set_ylabel(T("f12_ms", lang), fontsize=10.5, color=INK_SECONDARY)
        ax.legend(loc="upper left", frameon=False, fontsize=9.2)

    latency_panel(axes[0], "ttft_p50_ms", "ttft_p99_ms", "f12_ttft_title", logy=True)
    latency_panel(axes[1], "tpot_p50_ms", "tpot_p99_ms", "f12_tpot_title", logy=False)

    ax = axes[2]
    tp = [r["out_tok_per_s"] for r in rows]
    ax.bar(xs, tp, width=0.6, color=ROLE_COLOR["fp16"], zorder=3, edgecolor=SURFACE, linewidth=1.0)
    for x, v in zip(xs, tp):
        ax.text(x, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.6, color=INK, zorder=4)
    ax.set_title(T("tok_s", lang), fontsize=11.6, color=INK, pad=9, loc="left")
    ax.yaxis.set_major_formatter(_INT_FMT)
    ax.set_ylim(0, max(tp) * 1.18)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(xticklabels, fontsize=9.2, color=INK)
        ax.set_xlabel(T("f12_xlabel", lang), fontsize=10.5, color=INK_SECONDARY)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(labelsize=9.2)
        ax.set_facecolor(SURFACE)

    fig.suptitle(T("f12_suptitle", lang), fontsize=13.5, color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.01, 1, 0.91))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F11 -- RWKV-7 vs Qwen3.5, same engine (§13.1): the two bsz1 readings the doc
# tables draw a hard line between (deployment = each side's fastest config, so
# RWKV's fp16 hand kernels vs Qwen's bf16-only; architecture = both bf16 stock),
# plus the peak-concurrency reading on its own axis. Every bar is the c=1 (or
# sweep-peak) value from its own landed raw; the win % under each group is
# recomputed here from the two plotted bars, never lifted from the doc.
# ---------------------------------------------------------------------------
def f11_data():
    def b1(rp):
        return dict(load_rows(rp))[1]

    def pk(rp):
        c, v = sweep_peak(load_rows(rp))
        return v, c

    # explicit ordered list (no dict iteration) -> deterministic
    return [
        {
            "size": "1.5B / 2B",
            "rwkv_deploy": b1("w1prime_legEf_1.5b_5090.json"),
            "rwkv_bf16": b1("qwen35/rwkv7_1.5b_bf16_bsz1_5090.json"),
            "qwen_bf16": b1("qwen35/qwen35_2b_bf16_bsz1_5090.json"),
            "rwkv_peak": pk("qwen35/rwkv7_1.5b_bf16_sweep_5090.json"),
            "qwen_peak": pk("qwen35/qwen35_2b_bf16_sweep_5090.json"),
        },
        {
            "size": "7.2B / 9B",
            "rwkv_deploy": b1("w1prime_legFinal_B_7.2b_5090.json"),
            "rwkv_bf16": b1("qwen35/rwkv7_7.2b_bf16_bsz1_5090.json"),
            "qwen_bf16": b1("qwen35/qwen35_9b_bf16_bsz1_5090.json"),
            "rwkv_peak": pk("qwen35/rwkv7_7.2b_bf16_sweep_5090_v2.json"),
            "qwen_peak": pk("qwen35/qwen35_9b_bf16_sweep_5090_v2.json"),
        },
    ]


def fig_f11_qwen35_readings(out_path, lang):
    tiers = f11_data()
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(13.6, 6.6), dpi=DPI, gridspec_kw={"width_ratios": [1.55, 1.0]}
    )
    xs = list(range(len(tiers)))

    # ---- panel A: bsz1, three bars per tier (fp16-deploy / bf16-arch / qwen) ----
    w = 0.26
    for i, t in enumerate(tiers):
        bars = [
            (i - w, t["rwkv_deploy"], XCOLOR["ours"], T("f11_rwkv_deploy", lang)),
            (i, t["rwkv_bf16"], XCOLOR["ours_soft"], T("f11_rwkv_bf16", lang)),
            (i + w, t["qwen_bf16"], XCOLOR["qwen"], T("f11_qwen", lang)),
        ]
        for x, v, c, lab in bars:
            axa.bar(x, v, width=w, color=c, zorder=3, edgecolor=SURFACE, linewidth=1.0,
                    label=lab if i == 0 else None)
            axa.text(x, v, f" {v:,.1f}", rotation=90, ha="center", va="bottom",
                     fontsize=8.6, color=INK, zorder=4)
    top_a = max(max(t["rwkv_deploy"], t["rwkv_bf16"], t["qwen_bf16"]) for t in tiers)
    axa.set_ylim(0, top_a * 1.42)
    # per-group verdicts, recomputed from the plotted bars -> impossible to
    # misread which reading each margin belongs to
    for i, t in enumerate(tiers):
        dep = (t["rwkv_deploy"] / t["qwen_bf16"] - 1.0) * 100.0   # RWKV fp16 vs Qwen
        arch = (t["qwen_bf16"] / t["rwkv_bf16"] - 1.0) * 100.0    # Qwen vs RWKV bf16
        ytag = top_a * 1.30
        axa.text(i, ytag, T("f11_v_deploy", lang).format(p=dep), ha="center", va="bottom",
                 fontsize=9.4, color=XCOLOR["ours"], fontweight="bold", zorder=5)
        axa.text(i, ytag - top_a * 0.075, T("f11_v_arch", lang).format(p=arch), ha="center",
                 va="bottom", fontsize=9.4, color=XCOLOR["qwen"], fontweight="bold", zorder=5)
    axa.set_xticks(xs)
    axa.set_xticklabels([t["size"] for t in tiers], fontsize=11, color=INK)
    axa.set_title(T("f11_pa_title", lang), fontsize=12.2, color=INK, pad=9, loc="left")
    axa.set_ylabel(T("tok_s", lang), fontsize=11.5, color=INK_SECONDARY)

    # ---- panel B: peak concurrency, two bars per tier (both bf16) ----
    wb = 0.2
    for i, t in enumerate(tiers):
        rp_v, rp_c = t["rwkv_peak"]
        qp_v, qp_c = t["qwen_peak"]
        for x, v, c, cc in ((i - wb, rp_v, XCOLOR["ours_soft"], rp_c), (i + wb, qp_v, XCOLOR["qwen"], qp_c)):
            axb.bar(x, v, width=2 * wb, color=c, zorder=3, edgecolor=SURFACE, linewidth=1.0)
            axb.text(x, v, f" {v:,.0f}\n {T('f11_at', lang).format(c=cc)}", rotation=0, ha="center",
                     va="bottom", fontsize=8.2, color=INK, zorder=4)
    top_b = max(max(t["rwkv_peak"][0], t["qwen_peak"][0]) for t in tiers)
    axb.set_ylim(0, top_b * 1.28)
    for i, t in enumerate(tiers):
        peak = (t["rwkv_peak"][0] / t["qwen_peak"][0] - 1.0) * 100.0
        axb.text(i, top_b * 1.17, T("f11_v_peak", lang).format(p=peak), ha="center", va="bottom",
                 fontsize=9.4, color=XCOLOR["ours"], fontweight="bold", zorder=5)
    axb.set_xticks(xs)
    axb.set_xticklabels([t["size"] for t in tiers], fontsize=11, color=INK)
    axb.set_title(T("f11_pb_title", lang), fontsize=12.2, color=INK, pad=9, loc="left")
    axb.set_ylabel(T("tok_s", lang), fontsize=11.5, color=INK_SECONDARY)

    for ax in (axa, axb):
        ax.yaxis.set_major_formatter(_INT_FMT)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(labelsize=10.5)
        ax.tick_params(axis="x", length=0)
        ax.set_facecolor(SURFACE)

    handles, labels = axa.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               frameon=False, fontsize=10, handlelength=1.6)
    fig.suptitle(T("f11_suptitle", lang), fontsize=14.5, color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F13 -- §8 launch-autotune, kernel-level A/B per card per projection shape.
# Three GEMV projection shapes, NOT precision tiers: this figure has no tier
# axis and never shares a panel with a tier legend, so a single-hue blue ramp
# ("the same autotune, three shapes of the one kernel") avoids borrowing a
# tier's meaning. Each gain is recomputed from the two timed legs in the raw.
# ---------------------------------------------------------------------------
F13_SHAPES = [
    ("att_rkvo", "#a9c9ee"),
    ("ffn_key", "#5b97dd"),
    ("ffn_value", HUE["blue"]),
]


def f13_data():
    """§8's kernel-level A/B -- one gain per (card, projection shape), recomputed
    here as (heuristic_us / best_locked_us - 1)*100 straight from the two timed
    legs (heuristic = built-in launch heuristic, best_locked = our autotuned
    launch), never read off the raw's own `locked_gain_pct` or the doc table.
    9 cloud cards from autotune_ab_9cards.json + the workstation RTX 5090 from
    autotune_ab_5090.json. The RTX 3090's §8 zero is serving-level (no kernel A/B
    raw exists), so it is a caption note below, not a bar.
    """
    order = ["T4", "L4", "A10G", "A100-40GB", "A100-80GB", "L40S", "H100", "H200", "B200"]
    d9 = load_raw("autotune_ab_9cards.json")
    d5090 = load_raw("autotune_ab_5090.json")

    def gains(node):
        return [(node["shapes"][name]["heuristic_us"] / node["shapes"][name]["best_locked_us"] - 1.0)
                * 100.0 for name, _c in F13_SHAPES]

    cards = [{"name": k, "sm": d9[k]["sm"], "gains": gains(d9[k])} for k in order]
    cards.append({"name": "RTX 5090", "sm": d5090["sm"], "gains": gains(d5090)})
    cards.sort(key=lambda c: max(c["gains"]))  # ascending -> barh renders the max-gain card on top
    return cards


def fig_f13_autotune(out_path, lang):
    cards = f13_data()
    fig, ax = plt.subplots(figsize=(11.0, 8.4), dpi=DPI)

    n = len(cards)
    off = [0.26, 0.0, -0.26]  # att_rkvo top, ffn_value bottom within each card cluster
    for i, c in enumerate(cards):
        allzero = max(c["gains"]) < 0.05
        for (name, color), o, g in zip(F13_SHAPES, off, c["gains"]):
            ax.barh(i + o, g, height=0.24, color=color, zorder=3, edgecolor=SURFACE,
                    linewidth=0.6, label=name if i == n - 1 else None)
            ax.text(g + max(c["gains"], default=1) * 0.006 + 0.12, i + o, f"+{g:.1f}%",
                    va="center", ha="left", fontsize=7.6,
                    color=INK_MUTED if allzero else INK, zorder=4)

    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{c['name']}  (sm{c['sm']})" for c in cards], fontsize=9.6, color=INK)
    ax.set_ylim(-0.6, n - 0.4)
    xmax = max(max(c["gains"]) for c in cards)
    ax.set_xlim(0, xmax * 1.17)
    ax.set_title(T("f13_title", lang), fontsize=13, color=INK, pad=11, loc="left")
    ax.set_xlabel(T("f13_ylabel", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:g}%"))
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(labelsize=9.6)
    ax.tick_params(axis="y", length=0)
    ax.set_facecolor(SURFACE)

    # near-zero cards sit at the bottom (sorted asc), so the bottom-right corner is
    # empty -- anchor both honest-zero notes there.
    ax.text(0.985, 0.075, T("f13_zero_note", lang), transform=ax.transAxes, fontsize=8.6,
            color=INK_SECONDARY, ha="right", va="bottom", style="italic")
    ax.text(0.985, 0.028, T("f13_3090_note", lang), transform=ax.transAxes, fontsize=8.6,
            color=INK_MUTED, ha="right", va="bottom", style="italic")

    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _n, col in F13_SHAPES]
    ax.legend(handles, [nm for nm, _c in F13_SHAPES], title=T("f13_shape_legend", lang),
              loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3, frameon=False,
              fontsize=9.6, title_fontsize=9.6)
    fig.tight_layout(rect=(0, 0.05, 1, 0.99))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F14 -- §10 constant-state story. FORMULA-DERIVED, not measured: the one
# exception to this module's raw-only rule. It illustrates the per-request
# state-size equations printed in §13/F0056 (RWKV-7 state = L·(2D+64D),
# constant in context length; Qwen3.5-2B's 6 attention layers carry a KV cache
# that grows with T). The on-figure note and the doc caption both say so.
# ---------------------------------------------------------------------------
MIB = 2 ** 20


def f14_rwkv_state_mib(L, D, state_fp16=False):
    # token-shift states (2D elements) stay fp32; the WKV recurrent state
    # (64D elements = the [H,64,64] matrix, H=D/64) is the part RWKV_STATE_FP16
    # narrows to fp16. Reproduces §13/F0056's recorded constants: 7.2B 33.0/17.0
    # MiB, 1.5B 12.4 MiB (= the doc's "12.98 MB") / 6.4 MiB ("6.68 MB").
    shift_bytes = 2 * D * 4
    wkv_bytes = 64 * D * (2 if state_fp16 else 4)
    return L * (shift_bytes + wkv_bytes) / MIB


def f14_qwen_state_mib(T, L=24, D=2048):
    # Qwen3.5-2B (§13's printed formula): 18 GDN layers (¾·L) contribute a
    # constant state; its 6 attention layers (¼·L) carry a KV cache ∝ T.
    gdn_bytes = (L * 3 // 4) * (3 * 6 * D + 2 * 128 * D)     # constant offset
    kv_bytes = (L // 4) * (2 * 2 * 256 * T)                  # grows with context
    return (gdn_bytes + kv_bytes) / MIB


def fig_f14_state_vs_kv(out_path, lang):
    fig, ax = plt.subplots(figsize=(10.4, 6.6), dpi=DPI)
    T_MAX = 65536
    ctx = list(range(0, T_MAX + 1, 1024))

    # (constant MiB, color, linestyle, label key) -- explicit list, deterministic
    flat = [
        (f14_rwkv_state_mib(32, 4096), ROLE_COLOR["fp16"], "-", "f14_rwkv72"),
        (f14_rwkv_state_mib(32, 4096, state_fp16=True), ROLE_COLOR["fp16_state_fp16"], "--", "f14_rwkv_fp16state"),
        (f14_rwkv_state_mib(24, 2048), XCOLOR["ours_soft"], "-", "f14_rwkv15"),
    ]
    end_entries = []
    for yval, color, ls, key in flat:
        ax.plot([0, T_MAX], [yval, yval], linestyle=ls, linewidth=2, color=color, zorder=3,
                label=T(key, lang))
        end_entries.append((T_MAX, yval, color, f"{yval:.1f} MiB"))

    qy = [f14_qwen_state_mib(t) for t in ctx]
    ax.plot(ctx, qy, linestyle="-", linewidth=2.4, color=XCOLOR["qwen"], zorder=4,
            label=T("f14_qwen", lang))
    end_entries.append((T_MAX, qy[-1], XCOLOR["qwen"], f"{qy[-1]:.0f} MiB"))
    tmid = ctx[len(ctx) * 6 // 10]
    ax.annotate(T("f14_grow", lang), (tmid, f14_qwen_state_mib(tmid)), xytext=(-2, 13),
                textcoords="offset points", fontsize=9, color=XCOLOR["qwen"], rotation=20,
                ha="center", va="bottom", zorder=5, fontweight="bold")

    ax.set_xlim(0, T_MAX * 1.015)
    ax.set_ylim(0, max(qy) * 1.12)
    ax.xaxis.set_major_locator(FixedLocator([0, 16384, 32768, 49152, 65536]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: "0" if x == 0 else f"{x / 1024:g}K"))
    ax.yaxis.set_major_formatter(_INT_FMT)
    ax.set_xlabel(T("f14_xlabel", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.set_ylabel(T("f14_ylabel", lang), fontsize=11.5, color=INK_SECONDARY)
    ax.set_title(T("f14_title", lang), fontsize=13, color=INK, pad=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(labelsize=10.5)
    ax.set_facecolor(SURFACE)

    ax.legend(loc="upper left", frameon=False, fontsize=9.6, bbox_to_anchor=(0.02, 0.995))
    ax.text(0.035, 0.63, T("f14_note", lang), transform=ax.transAxes, fontsize=8.2,
            color=INK_SECONDARY, va="top", ha="left", linespacing=1.5, style="italic")

    fig.tight_layout(rect=(0, 0.01, 0.9, 0.99))
    fig.canvas.draw()  # finalize transforms before pixel-space end-label math
    _end_labels(ax, end_entries)
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F15 -- §4 quantization tradeoff, two complementary panels that F3 (the 7.2B
# accuracy/speed frontier) and F4 (the MATH500 ladder) do NOT already cover:
#   A: compression-rate Δ vs same-size fp16 across tiers AND both sizes -- the
#      one accuracy metric int8 w8g64/w8a8 has a landed 7.2B number for, and the
#      metric §4 says "hides int4's reasoning damage".
#   B: 1.5B single-stream speed vs MATH500 avg@64, marker area = weight VRAM --
#      the size where int4's MATH500 collapse is dramatic (so it complements
#      F3's 7.2B frontier instead of duplicating it). int4's accuracy raw is a
#      3090 run (F0043-era, per F4's own note) while its speed is 5090; that
#      one cross-card pair is labeled on-figure, F3-style.
# ---------------------------------------------------------------------------
def f15a_data():
    fp16_15 = pooled_bpb("uncheatable_full_fp16_1.5b_5090main.json")
    fp16_72 = pooled_bpb("uncheatable_full_fp16_7.2b_5090main.json")
    return [
        ("f15_size_15", [
            ("w8g64", pooled_bpb("uncheatable_full_w8_1.5b_5090main.json") - fp16_15),
            ("w8a8", pooled_bpb("uncheatable_full_w8a8_1.5b_5090main.json") - fp16_15),
            ("int4_gptq", pooled_bpb("uncheatable_full_w4_1.5b_5090main.json") - fp16_15),
        ]),
        # 7.2B has no landed w8g64 uncheatable raw -> that tier slot is left empty
        # (honest gap), not back-filled.
        ("f15_size_72", [
            ("w8a8", pooled_bpb("uncheatable_full_w8a8_7.2b_5090main.json") - fp16_72),
            ("int4_gptq", pooled_bpb("uncheatable_full_w4_7.2b_5090main.json") - fp16_72),
        ]),
    ]


def f15b_data():
    # 1.5B total params (§13's non-emb table: total 1.527B). Weight footprint is
    # the standard params x byte-width proxy (fp16 2B, int8 1B, int4 0.5B/param),
    # stated as indicative in the caption -- §4 prints a weight-GB figure only for
    # 7.2B int4 (4.6 GB), and this panel is 1.5B.
    PARAMS = 1.527e9

    def c1(p):
        return dict(load_rows(p))[1]

    return [
        {"role": "fp16", "speed": c1("bsz_sweep_1.5b_fp16_5090.json"),
         "acc": math500_pct("math500_avg64_5090main.json")[0], "gb": PARAMS * 2 / 1e9, "cross": False},
        {"role": "w8a8", "speed": c1("bsz_sweep_w8a8v2_5090main.json"),
         "acc": math500_pct("math500_avg64_w8a8_5090main.json")[0], "gb": PARAMS * 1 / 1e9, "cross": False},
        {"role": "int4_gptq", "speed": c1("bsz_sweep_1.5b_w4gptq_5090.json"),
         "acc": math500_pct("math500_avg64_1.5b_sym.json")[0], "gb": PARAMS * 0.5 / 1e9, "cross": True},
    ]


def fig_f15_quant_tradeoff(out_path, lang):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.8, 6.2), dpi=DPI,
                                   gridspec_kw={"width_ratios": [1.0, 1.12]})

    # ---- panel A: compression-rate Δ, grouped by size, tier-hued ----
    slot = {"w8g64": -0.27, "w8a8": 0.0, "int4_gptq": 0.27}  # fixed tier slots
    clusters = f15a_data()
    seen = []
    for ci, (size_key, tiers) in enumerate(clusters):
        for role, delta in tiers:
            axa.bar(ci + slot[role], delta, width=0.25, color=ROLE_COLOR[role], zorder=3,
                    edgecolor=SURFACE, linewidth=1.0,
                    label=role_label(role, lang) if role not in seen else None)
            seen.append(role)
            axa.text(ci + slot[role], delta, f"+{delta:.4f}", ha="center", va="bottom",
                     fontsize=8.4, color=INK, zorder=4)
    max_d = max(d for _s, ts in clusters for _r, d in ts)
    axa.set_ylim(0, max_d * 1.22)
    axa.set_xticks(range(len(clusters)))
    axa.set_xticklabels([T(s, lang) for s, _t in clusters], fontsize=11, color=INK)
    axa.set_title(T("f15a_title", lang), fontsize=10.8, color=INK, pad=9, loc="left")
    axa.set_ylabel(T("f15a_ylabel", lang), fontsize=11, color=INK_SECONDARY)
    axa.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f"{y:.3f}"))
    axa.text(0.5, -0.14, T("f15a_note", lang), transform=axa.transAxes, fontsize=8.4,
             color=INK_MUTED, ha="center", va="top", style="italic")
    axa.legend(loc="upper left", frameon=False, fontsize=9.6)

    # ---- panel B: 1.5B speed vs MATH500, marker area = weight VRAM ----
    pts = f15b_data()
    for p in pts:
        axb.scatter([p["speed"]], [p["acc"]], s=130 * p["gb"], color=ROLE_COLOR[p["role"]],
                    edgecolors=SURFACE, linewidths=1.3, zorder=4)
    # annotations placed away from the markers; explicit per-point offsets (3 points)
    ann = {"fp16": (10, 20, "left"), "w8a8": (12, 18, "left"), "int4_gptq": (10, 24, "left")}
    for p in pts:
        dx, dy, ha = ann[p["role"]]
        txt = f"{role_label(p['role'], lang)}\n{p['speed']:,.0f} tok/s · {p['acc']:.1f}% · ~{p['gb']:.1f} GB"
        if p["cross"]:
            txt += f"\n{T('f15b_cross', lang)}"
        axb.annotate(txt, (p["speed"], p["acc"]), textcoords="offset points", xytext=(dx, dy),
                     fontsize=8.2, color=INK, zorder=6, ha=ha, va="bottom", linespacing=1.4,
                     arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.7))
    sp = [p["speed"] for p in pts]
    ac = [p["acc"] for p in pts]
    axb.set_xlim(min(sp) - 60, max(sp) + 90)
    axb.set_ylim(min(ac) - 8, max(ac) + 16)
    axb.set_title(T("f15b_title", lang), fontsize=10.8, color=INK, pad=9, loc="left")
    axb.set_xlabel(T("f15b_xlabel", lang), fontsize=11, color=INK_SECONDARY)
    axb.set_ylabel(T("math500_ylabel", lang), fontsize=11, color=INK_SECONDARY)
    axb.xaxis.set_major_formatter(_INT_FMT)
    axb.text(0.5, -0.145, T("f15_vram_note", lang), transform=axb.transAxes, fontsize=8.4,
             color=INK_MUTED, ha="center", va="top", style="italic")

    for ax in (axa, axb):
        ax.grid(True, axis="y" if ax is axa else "both", color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(labelsize=10)
        ax.set_facecolor(SURFACE)
    axa.tick_params(axis="x", length=0)

    fig.suptitle(T("f15_suptitle", lang), fontsize=14.5, color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


FIGURES = [
    ("f1_concurrency_5090", fig_f1_concurrency_5090),
    ("f2_concurrency_3090", fig_f2_concurrency_3090),
    ("f3_accuracy_speed_frontier", fig_f3_accuracy_speed_frontier),
    ("f4_math500_ladder", fig_f4_math500_ladder),
    ("f6_fleet", fig_f6_fleet),
    ("f7_speed_ladder", fig_f7_speed_ladder),
    ("f8_albatross", fig_f8_albatross),
    ("f9a_sharegpt_engines", fig_f9a_sharegpt_engines),
    ("f9b_sharegpt_w4", fig_f9b_sharegpt_w4),
    ("f10_tp_pp", fig_f10_tp_pp),
    ("f11_qwen35_readings", fig_f11_qwen35_readings),
    ("f12_latency_poisson", fig_f12_latency_poisson),
    ("f13_autotune", fig_f13_autotune),
    ("f14_state_vs_kv", fig_f14_state_vs_kv),
    ("f15_quant_tradeoff", fig_f15_quant_tradeoff),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Manifest check (raw files each figure consumes):")
    for fig_id, paths in MANIFEST.items():
        missing = [p for p in paths if not exists(p)]
        status = "OK" if not missing else f"MISSING {missing}"
        print(f"  {fig_id}: {len(paths)} raw(s) -- {status}")

    for lang in LANGS:
        _set_lang_fonts(lang)
        print(f"-- lang={lang} --")

        for base_name, fn in FIGURES:
            out_path = os.path.join(OUT_DIR, f"{base_name}{_suffix(lang)}.svg")
            fn(out_path, lang)
            print(f"wrote {os.path.relpath(out_path, REPO_ROOT)}")

        f5_path = os.path.join(OUT_DIR, f"f5_positional_compression_state_precision{_suffix(lang)}.svg")
        if fig_f5_positional_compression_state_precision(f5_path, lang):
            print(f"wrote {os.path.relpath(f5_path, REPO_ROOT)}")
        else:
            print("SKIPPED f5_positional_compression_state_precision{,_zh}.svg -- raws not landed yet "
                  "(uncheatable_positional_7.2b_fp16_state{32,16}_3090.json); see MANIFEST.")


if __name__ == "__main__":
    main()
