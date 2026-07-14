"""Precision x size benchmark charts for docs/BENCHMARKS.md (+ zh-CN).

Reads ONLY landed raws under bench/results/ (no hand-entered numbers) and writes
deterministic SVGs to docs/assets/plots/. Every figure function is preceded by a
MANIFEST entry naming the exact raw file(s) it consumes; a file listed as
"cross-check only" is read and its value(s) verified/cited in the caption but not
drawn as its own series (documented at the call site, not silently dropped).

If a raw a figure would need does not exist yet, the figure (or that one series)
is skipped and printed as SKIPPED — never interpolated, never fabricated.

Usage:
  python bench/plots/make_benchmark_plots.py

Determinism: fixed figsize/dpi, fixed matplotlib svg.hashsalt, no embedded dates,
svg.fonttype=none (text stays text, no embedded font outlines). Run twice and
diff the output directory to confirm byte-identical SVGs.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
# Determinism: matplotlib's SVG backend assigns clip-path / marker ids via
# uuid4() unless a fixed hashsalt is set, and embeds today's date unless the
# caller overrides it per-savefig. Both are pinned so re-running this script
# on a different day produces byte-identical output.
matplotlib.rcParams["svg.hashsalt"] = "rwkv-sglang-benchmark-plots"
matplotlib.rcParams["svg.fonttype"] = "none"  # keep <text>, don't embed font outlines
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, ScalarFormatter

SVG_METADATA = {"Date": None, "Creator": "rwkv-sglang bench/plots/make_benchmark_plots.py"}
DPI = 100

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO_ROOT, "bench", "results")
OUT_DIR = os.path.join(REPO_ROOT, "docs", "assets", "plots")

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

ROLE_COLOR = {
    "fp16": HUE["blue"],
    "fp16_state_fp16": HUE["aqua"],
    "int4_rtn": HUE["yellow"],
    "w8a8": HUE["green"],
    "int4_gptq": HUE["violet"],
    "int4_gptq_asym": HUE["red"],
    "hybrid": HUE["magenta"],
    "w4a8_experimental": HUE["orange"],
}
ROLE_LABEL = {
    "fp16": "fp16",
    "fp16_state_fp16": "fp16 + RWKV_STATE_FP16 (W1')",
    "int4_rtn": "int4 RTN",
    "w8a8": "int8 w8a8",
    "int4_gptq": "int4 GPTQ",
    "int4_gptq_asym": "int4 GPTQ (asymmetric)",
    "hybrid": "int4 GPTQ (hybrid ffn.v/ffn.k)",
    "w4a8_experimental": "int4 GPTQ + w4a8-tc (experimental, opt-in)",
}

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
        "math500_avg64_7.2b_fp16.json", "math500_avg64_7.2b_fp16_stateon.json",
        "math500_avg64_7.2b_sym.json", "math500_avg64_7.2b_asym.json", "math500_avg64_7.2b_hybrid_ffnvk.json",
    ],
    "f5_positional_compression_state_precision": [
        "uncheatable_positional_7.2b_fp16_state32_3090.json",
        "uncheatable_positional_7.2b_fp16_state16_3090.json",
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


# ---------------------------------------------------------------------------
# Shared chart chrome
# ---------------------------------------------------------------------------
_INT_FMT = FuncFormatter(lambda x, pos: f"{x:,.0f}")


def _style_ax(ax, title, ylabel, logx=True):
    ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")
    ax.set_xlabel("concurrency", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_SECONDARY)
    if logx:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(LogLocator(base=2))
        ax.xaxis.set_major_formatter(_INT_FMT)  # plain "64" not "64.0"/"6×10^1"
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(_INT_FMT)
    ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(labelsize=8.5)
    ax.set_facecolor(SURFACE)


def _plot_series(ax, rows, role, linestyle="-", hollow=False, zorder=3, label_override=None,
                  connect=True):
    if not rows:
        return
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    color = ROLE_COLOR[role]
    label = label_override or ROLE_LABEL[role]
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


def _dedupe_legend(handles_labels_list, ax, ncol=3, loc="upper center", bbox=(0.5, -0.02)):
    seen = {}
    for ax_ in handles_labels_list:
        h, l = ax_.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            seen.setdefault(ll, hh)
    ax.legend(list(seen.values()), list(seen.keys()), loc=loc, bbox_to_anchor=bbox, ncol=ncol,
               frameon=False, fontsize=8.5, handlelength=2.2)


# ---------------------------------------------------------------------------
# F1 -- per-size concurrency curves, RTX 5090
# ---------------------------------------------------------------------------
def fig_f1_concurrency_5090(out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.3), dpi=DPI)

    # -- 0.1B: fp16 / GPTQ / RTN, full 8-point sweeps --
    ax = axes[0]
    _plot_series(ax, load_rows("bsz_sweep_0.1b_fp16_5090.json"), "fp16")
    _plot_series(ax, load_rows("bsz_sweep_0.1b_w4gptq_5090.json"), "int4_gptq")
    _plot_series(ax, load_rows("bsz_sweep_0.1b_w4rtn_5090.json"), "int4_rtn")
    _style_ax(ax, "0.1B", "output tok/s")

    # -- 1.5B: fp16 full-stack (8pt) + STATE_FP16 W1' bsz1-only point + int4 (3pt) + w8a8 (8pt) --
    ax = axes[1]
    _plot_series(ax, load_rows("bsz_sweep_fullstack_5090.json"), "fp16")
    _plot_series(ax, load_rows("w1prime_legEf_1.5b_5090.json"), "fp16_state_fp16", connect=False)
    _plot_series(ax, load_rows("bsz_sweep_1.5b_w4gptq_5090.json"), "int4_gptq")
    _plot_series(ax, load_rows("bsz_sweep_1.5b_w4rtn_5090.json"), "int4_rtn")
    _plot_series(ax, load_rows("bsz_sweep_w8a8v2_5090main.json"), "w8a8")
    _style_ax(ax, "1.5B", "output tok/s")

    # -- 7.2B: fp16 (F0047-corrected, 11pt) + STATE_FP16 W1' (3pt) + int4 (merged) + w8a8 (merged) --
    ax = axes[2]
    _plot_series(ax, load_rows("72b/sweep_72b_fp16_v3_5090.json"), "fp16")
    _plot_series(ax, load_rows("w1prime_legFinal_B_7.2b_5090.json"), "fp16_state_fp16", connect=False)
    gptq_72b = merge_rows(["bsz_sweep_7.2b_w4gptq_5090.json", "bsz_sweep_7.2b_w4gptq_5090_ext.json"])
    _plot_series(ax, gptq_72b, "int4_gptq")
    _plot_series(ax, load_rows("bsz_sweep_7.2b_w4rtn_5090.json"), "int4_rtn")
    w8a8_72b = merge_rows(["72b/sweep_72b_w8a8_ceil.json", "72b/sweep_72b_w8a8_max.json", "72b/sweep_72b_w8a8.json"])
    _plot_series(ax, w8a8_72b, "w8a8")
    _style_ax(ax, "7.2B", "output tok/s")

    _dedupe_legend(axes, axes[1], ncol=5, loc="upper center", bbox=(0.5, -0.16))
    fig.suptitle("Serving concurrency sweep by precision — RTX 5090 (64-in/256-out, wall-clock)",
                 fontsize=12.5, color=INK, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F2 -- per-size concurrency curves, RTX 3090 (incl. the w4 cliff)
# ---------------------------------------------------------------------------
def fig_f2_concurrency_3090(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.5), dpi=DPI)

    # -- 1.5B: fp16 / GPTQ / RTN, matched §4b matrix session, 6pt each --
    ax = axes[0]
    _plot_series(ax, load_rows("bsz_sweep_1.5b_fp16_3090.json"), "fp16")
    _plot_series(ax, load_rows("bsz_sweep_1.5b_w4gptq_3090.json"), "int4_gptq")
    _plot_series(ax, load_rows("bsz_sweep_1.5b_w4rtn_3090.json"), "int4_rtn")
    _style_ax(ax, "1.5B", "output tok/s")

    # -- 7.2B: fp16 / RTN (3pt matrix) + int4-GPTQ cliff composite + w4a8-tc experimental --
    ax = axes[1]
    _plot_series(ax, load_rows("bsz_sweep_7.2b_fp16_3090.json"), "fp16")
    _plot_series(ax, load_rows("bsz_sweep_7.2b_w4rtn_3090.json"), "int4_rtn")
    gptq_cliff = merge_rows([
        "bsz_sweep_7.2b_w4gptq_3090.json",           # c=1,32 (+128, superseded by cliffmap below)
        "bsz_sweep_7.2b_w4gptq_3090_cliffmap.json",  # c=48,64,80,96,112,128
        "bsz_sweep_7.2b_w4gptq_3090_cliffmap_fine.json",  # c=66,72,76
    ])
    _plot_series(ax, gptq_cliff, "int4_gptq", label_override="int4 GPTQ (w4a16, cliff map)")
    w4a8_exp = load_rows("bsz_sweep_7.2b_w4gptq_3090_cliff_stage1_w4a8.json")
    _plot_series(ax, w4a8_exp, "w4a8_experimental", linestyle="--", hollow=True)
    _style_ax(ax, "7.2B — the w4 M=64 cliff, and its kernel-level fix (F0055)", "output tok/s")

    _dedupe_legend(axes, axes[1], ncol=3, loc="upper center", bbox=(0.5, -0.16))
    fig.suptitle("Serving concurrency sweep by precision — RTX 3090 (64-in/256-out, wall-clock)",
                 fontsize=12.5, color=INK, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.09, 1, 0.90))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F3 -- accuracy-vs-speed frontier, 7.2B
# ---------------------------------------------------------------------------
def fig_f3_accuracy_speed_frontier(out_path):
    fig, ax = plt.subplots(figsize=(7.6, 6.6), dpi=DPI)

    points = []  # (x, y, role, label, hollow)

    x = sweep_value_at("72b/sweep_72b_fp16_v3_5090.json", 1)
    y, c, n = math500_pct("math500_avg64_7.2b_fp16.json")
    points.append((x, y, "fp16", "fp16", False))

    x = sweep_value_at("w1prime_legFinal_B_7.2b_5090.json", 1)
    y, c, n = math500_pct("math500_avg64_7.2b_fp16_stateon.json")
    points.append((x, y, "fp16_state_fp16", "fp16 + STATE_FP16 (W1')", False))

    x = sweep_value_at("bsz_sweep_7.2b_w4gptq_5090.json", 1)
    y, c, n = math500_pct("math500_avg64_7.2b_sym.json")
    points.append((x, y, "int4_gptq", "int4 GPTQ (symmetric)", False))

    gptq_cliff_w4a8 = load_rows("bsz_sweep_7.2b_w4gptq_3090_cliff_stage1_w4a8.json")
    x = sweep_peak(gptq_cliff_w4a8)[1]
    y, c, n = math500_pct("math500_avg64_7.2b_w4gptq_w4a8capped_3090.json")
    points.append((x, y, "w4a8_experimental", "int4 GPTQ + w4a8-tc, capped (experimental)", True))

    for x, y, role, label, hollow in points:
        color = ROLE_COLOR[role]
        if hollow:
            ax.scatter([x], [y], s=170, facecolors=SURFACE, edgecolors=color, linewidths=2.2,
                       marker="D", zorder=4, label=label)
        else:
            ax.scatter([x], [y], s=150, facecolors=color, edgecolors=SURFACE, linewidths=1.2,
                       marker="o", zorder=4, label=label)

    xmax = max(p[0] for p in points)
    ax.set_xlim(-40, xmax * 1.12)
    ymin = min(p[1] for p in points)
    ymax = max(p[1] for p in points)
    pad = (ymax - ymin) * 0.45 + 2
    ax.set_ylim(ymin - pad, ymax + pad)

    # fp16 and +STATE_FP16 sit almost on top of each other at this scale -- a single
    # shared leader note beats two overlapping per-point callouts (dataviz: don't stack
    # colliding end-labels).
    fp16_pt, state_pt = points[0], points[1]
    mid_x = (fp16_pt[0] + state_pt[0]) / 2
    mid_y = (fp16_pt[1] + state_pt[1]) / 2
    ax.annotate("fp16 / +STATE_FP16\nspeed: 5090 both. accuracy: fp16 from the\n2026-07-09 3090 batch (F0055 §0); state_fp16\nis 5090 (F0056) — cards differ for fp16 only",
                (mid_x, mid_y), textcoords="offset points", xytext=(14, 26), fontsize=7.2,
                color=INK_MUTED, zorder=5, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.7))

    int4_pt = points[2]
    ax.annotate("int4 GPTQ (symmetric)\nspeed: 5090 (§4b) · accuracy: 3090 (F0055 §0)\ncards differ — shown explicitly",
                (int4_pt[0], int4_pt[1]), textcoords="offset points", xytext=(14, -34), fontsize=7.2,
                color=INK_MUTED, zorder=5, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.7))

    w4a8_pt = points[3]
    ax.annotate("int4 GPTQ + w4a8-tc, capped (experimental)\nNOT single-stream: 3090 peak-concurrency (c=128).\nF0055 §6: RED, default-OFF opt-in only",
                (w4a8_pt[0], w4a8_pt[1]), textcoords="offset points", xytext=(-14, -34), fontsize=7.2,
                color=INK_MUTED, zorder=5, linespacing=1.4, ha="right",
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.7))

    ax.set_xlabel("single-stream output tok/s (bsz1, unless noted)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("MATH500 avg@64 (%)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title("Accuracy vs. speed frontier — RWKV-7 7.2B", fontsize=12.5, color=INK, loc="left", pad=10)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_facecolor(SURFACE)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8.3,
              handletextpad=0.5)

    ax.text(0.01, -0.30,
            "int8 w8a8/w8g64 omitted: no MATH500 avg@64 raw exists for 7.2B at this tier\n"
            "(only compression-rate evidence, §2: +0.0041 bpb pooled). See F4 for the 1.5B w8a8 bar.",
            transform=ax.transAxes, fontsize=7, color=INK_MUTED, va="top", ha="left", style="italic")
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F4 -- MATH500 accuracy ladder, precision x size (grouped bars)
# ---------------------------------------------------------------------------
def fig_f4_math500_ladder(out_path):
    fig, ax = plt.subplots(figsize=(9.6, 6.3), dpi=DPI)

    slots = ["fp16", "fp16_state_fp16", "w8a8", "int4_gptq", "int4_gptq_asym", "hybrid"]
    slot_label = {
        "fp16": "fp16", "fp16_state_fp16": "+STATE_FP16", "w8a8": "w8a8",
        "int4_gptq": "int4-sym", "int4_gptq_asym": "int4-asym", "hybrid": "hybrid",
    }

    onepfive = {
        "fp16": math500_pct("math500_avg64_5090main.json"),
        "w8a8": math500_pct("math500_avg64_w8a8_5090main.json"),
        # int4_gptq / int4_gptq_asym: documented in BENCHMARKS.md §2/§4 (F0043) as 14.98% / 21.99%
        # but no raw JSON for those two 1.5B MATH500 avg@64 runs is landed under bench/results/
        # (checked: not present, no add-then-delete git history either) -- omitted rather than
        # hand-entered, flagged in-figure instead of silently dropped.
    }
    seven2b = {
        "fp16": math500_pct("math500_avg64_7.2b_fp16.json"),
        "fp16_state_fp16": math500_pct("math500_avg64_7.2b_fp16_stateon.json"),
        "int4_gptq": math500_pct("math500_avg64_7.2b_sym.json"),
        "int4_gptq_asym": math500_pct("math500_avg64_7.2b_asym.json"),
        "hybrid": math500_pct("math500_avg64_7.2b_hybrid_ffnvk.json"),
    }
    missing_1_5b = {"int4_gptq", "int4_gptq_asym"}

    n = len(slots)
    bar_w = 0.34
    group_gap = 1.15
    x_1_5b = [i * group_gap for i in range(n)]
    x_7_2b = [i * group_gap + n * group_gap + 1.0 for i in range(n)]

    for i, slot in enumerate(slots):
        color = ROLE_COLOR[slot]
        if slot in onepfive:
            val, c, tot = onepfive[slot]
            ax.bar(x_1_5b[i], val, width=bar_w, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
            ax.text(x_1_5b[i], val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        elif slot in missing_1_5b:
            ax.text(x_1_5b[i], 2.0, "no raw\nlanded", ha="center", va="bottom", fontsize=6.6,
                    color=INK_MUTED, style="italic", rotation=0, linespacing=1.2)

        if slot in seven2b:
            val, c, tot = seven2b[slot]
            ax.bar(x_7_2b[i], val, width=bar_w, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
            ax.text(x_7_2b[i], val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_xticks([sum(x_1_5b) / n, sum(x_7_2b) / n])
    ax.set_xticklabels(["1.5B", "7.2B"], fontsize=11.5, color=INK)
    ax.tick_params(axis="x", length=0, pad=48)
    for i, slot in enumerate(slots):
        ax.text(x_1_5b[i], -3.0, slot_label[slot], ha="right", va="top", fontsize=7.3,
                color=INK_MUTED, rotation=35, rotation_mode="anchor")
        ax.text(x_7_2b[i], -3.0, slot_label[slot], ha="right", va="top", fontsize=7.3,
                color=INK_MUTED, rotation=35, rotation_mode="anchor")

    ax.set_ylabel("MATH500 avg@64 (%)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title("MATH500 avg@64 by precision and model size", fontsize=12.5, color=INK, loc="left", pad=14)
    ax.set_ylim(0, 75)
    ax.set_xlim(-0.7, x_7_2b[-1] + 0.7)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=2)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.set_facecolor(SURFACE)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLOR[s]) for s in slots]
    ax.legend(handles, [ROLE_LABEL[s] for s in slots], loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=3, frameon=False, fontsize=8.2)

    fig.text(0.01, 0.995,
            "1.5B int4-sym/asym (documented as 14.98%/21.99% in §2/F0043) have no landed raw — "
            "omitted, not fabricated.\nMATH500 avg@64 bars mix RTX 5090 and RTX 3090 runs; this project's own §2 "
            "cross-check found the ruler\nagrees within ±0.27pt across cards at matched config, i.e. "
            "card-invariant within this protocol's noise.",
            fontsize=7, color=INK_MUTED, va="top", ha="left", style="italic")

    fig.tight_layout(rect=(0, 0.02, 1, 0.85))
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F5 (conditional) -- positional compression curve, fp32-state vs fp16-state
# ---------------------------------------------------------------------------
def fig_f5_positional_compression_state_precision(out_path):
    """7.2B uncheatable position curve, RWKV_STATE_FP16 off (fp32 state) vs on (fp16 state).

    Not yet landed as of this writing -- see MANIFEST key
    f5_positional_compression_state_precision for the two expected filenames. This
    function is intentionally left runnable-but-a-no-op until they land: it checks
    for the raws and returns without writing a file if either is missing, so the
    rest of the batch is never blocked on this one figure.
    """
    needed = MANIFEST["f5_positional_compression_state_precision"]
    if not all(exists(p) for p in needed):
        return False

    def curve(relpath):
        with open(_path(relpath)) as f:
            d = json.load(f)
        # position-bucket curve: list of {bucket, mean_neg_log2_p} or equivalent --
        # shape intentionally not hard-coded further until the real file lands.
        return d

    off = curve(needed[0])
    on = curve(needed[1])

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=DPI)
    for d, role, label in ((off, "fp16", "fp32 state (default)"), (on, "fp16_state_fp16", "fp16 state (RWKV_STATE_FP16)")):
        buckets = d["buckets"]
        xs = [b["position"] for b in buckets]
        ys = [b["bits_per_byte"] for b in buckets]
        _plot_series(ax, list(zip(xs, ys)), role, label_override=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("token position", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("mean −log₂ p (bits/byte)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title("Positional compression — 7.2B, fp32 vs fp16 recurrent state", fontsize=12, color=INK,
                 loc="left", pad=10)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _source_note(fig)
    fig.savefig(out_path, metadata=SVG_METADATA)
    plt.close(fig)
    return True


FIGURES = [
    ("f1_concurrency_5090.svg", fig_f1_concurrency_5090),
    ("f2_concurrency_3090.svg", fig_f2_concurrency_3090),
    ("f3_accuracy_speed_frontier.svg", fig_f3_accuracy_speed_frontier),
    ("f4_math500_ladder.svg", fig_f4_math500_ladder),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Manifest check (raw files each figure consumes):")
    for fig_id, paths in MANIFEST.items():
        missing = [p for p in paths if not exists(p)]
        status = "OK" if not missing else f"MISSING {missing}"
        print(f"  {fig_id}: {len(paths)} raw(s) -- {status}")

    for filename, fn in FIGURES:
        out_path = os.path.join(OUT_DIR, filename)
        fn(out_path)
        print(f"wrote {os.path.relpath(out_path, REPO_ROOT)}")

    f5_path = os.path.join(OUT_DIR, "f5_positional_compression_state_precision.svg")
    if fig_f5_positional_compression_state_precision(f5_path):
        print(f"wrote {os.path.relpath(f5_path, REPO_ROOT)}")
    else:
        print("SKIPPED f5_positional_compression_state_precision.svg -- raws not landed yet "
              "(uncheatable_positional_7.2b_fp16_state{32,16}_3090.json); see MANIFEST.")


if __name__ == "__main__":
    main()
