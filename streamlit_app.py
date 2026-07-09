"""PDA-Peak-Finder — Streamlit web app.

Interactive front-end for the analysis pipeline: load Waters Empower ARW
(or run the built-in synthetic demo), detect peaks on the MaxPlot, compute
RT / FWHM / lambda_max, drop peaks that barely absorb at a monitoring
wavelength (default 230 nm), and visualise everything in a Waters QDa/SIR
style labelled chromatogram, plus UV spectra, a contour map, tracking across
injections, and CSV downloads.

Run:
    python -m pip install -e ".[app]"
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pda_peak_finder import __version__
from pda_peak_finder.models import PDAData
from pda_peak_finder.peak_detection import (
    DeconvolutionConfig,
    PeakDetectionConfig,
    detect_peaks,
    detect_peaks_deconvolved,
    filter_peaks_by_height,
)
from pda_peak_finder.peak_detection.deconvolution import compute_baseline
from pda_peak_finder.plotting import (
    configure_japanese_font,
    plot_contour,
    plot_deconvolution,
    plot_labeled_chromatogram,
    plot_tracking,
    plot_uv_spectra,
)
from pda_peak_finder.models import Chromatogram
from pda_peak_finder.reader import load
from pda_peak_finder.spectra import (
    absorbance_at,
    annotate_peaks,
    filter_peaks_by_absorbance,
)
from pda_peak_finder.tracking import TrackingConfig, track_peaks
from pda_peak_finder.export import regression_table
from pda_peak_finder.testing import synthetic_pdadata

configure_japanese_font()

st.set_page_config(
    page_title="PDA-Peak-Finder",
    page_icon="🧪",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"


# --------------------------------------------------------------------------
# Data loading (cached — parsing a 70 MB ARW takes a few seconds)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="ARW を読み込み中…")
def _load_from_bytes(name: str, raw: bytes) -> PDAData:
    with tempfile.NamedTemporaryFile(suffix="_" + name, delete=False) as fh:
        fh.write(raw)
        tmp = fh.name
    return load(tmp)


@st.cache_data(show_spinner="ARW を読み込み中…")
def _load_from_path(path: str, _mtime: float) -> PDAData:
    return load(path)


def _decimate(data: PDAData, max_scans: int = 1500) -> PDAData:
    """Down-sample the time axis for a responsive contour plot."""
    step = max(1, len(data.times) // max_scans)
    if step == 1:
        return data
    return PDAData(
        times=data.times[::step],
        wavelengths=data.wavelengths,
        absorbance=data.absorbance[::step],
        metadata=data.metadata,
    )


# --------------------------------------------------------------------------
# Analysis (fast — recomputed live as sliders move)
# --------------------------------------------------------------------------
def analyse(data: PDAData, params: dict):
    if params["base_wl"] is not None:
        base = data.chromatogram_at(params["base_wl"], params["base_bw"])
    else:
        base = data.maxplot(wavelength_range=params["wl_range"])
    components = None
    if params["deconvolve"]:
        table, components = detect_peaks_deconvolved(
            base,
            DeconvolutionConfig(
                model=params["decon_model"],
                min_prominence=params["min_prominence"],
                min_distance_min=params["min_distance"],
            ),
            return_components=True,
        )
    else:
        table = detect_peaks(
            base,
            PeakDetectionConfig(
                min_height=params["min_height"],
                min_prominence=params["min_prominence"],
                min_distance_min=params["min_distance"],
            ),
        )
    table.source_label = base.label
    annotate_peaks(data, table)
    n_before = len(table)
    if params["monitor_on"]:
        table = filter_peaks_by_absorbance(
            table, params["monitor_wl"], min_absorbance=params["monitor_min_abs"]
        )
    if params["height_on"]:
        table = filter_peaks_by_height(
            table,
            min_height=params["height_min_au"],
            max_height=params["height_max_au"],
        )
    return base, table, n_before, components


def peak_dataframe(table, monitor_wl: float, uv_scale: float, height_unit: str) -> pd.DataFrame:
    rows = []
    for p in table:
        a_mon = absorbance_at(p.spectrum, monitor_wl) if p.spectrum is not None else np.nan
        row = {
            "peak_id": p.peak_id,
            "RT (min)": round(p.apex_time, 4),
            "FWHM (min)": round(p.fwhm, 4) if p.fwhm else None,
            "λmax (nm)": round(p.lambda_max, 1) if p.lambda_max else None,
            f"A@{monitor_wl:.0f}nm (AU)": round(float(a_mon), 5),
            "height (AU)": round(p.height, 5),
            "area (AU·min)": round(p.area, 6) if p.area else None,
        }
        if height_unit != "AU":
            row[f"height ({height_unit})"] = round(p.height * uv_scale, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sidebar — inputs and parameters
# --------------------------------------------------------------------------
st.sidebar.title("🧪 PDA-Peak-Finder")
st.sidebar.caption(f"v{__version__} · Waters Empower PDA/UV 3D 解析")

source = st.sidebar.radio(
    "データソース", ["サンプル (data/)", "アップロード", "合成デモ"], index=0
)

datasets: list[PDAData] = []
if source == "サンプル (data/)":
    available = sorted(DATA_DIR.glob("*.arw")) if DATA_DIR.exists() else []
    if not available:
        st.sidebar.info("data/ に .arw がありません。アップロードか合成デモを使ってください。")
    picked = st.sidebar.multiselect(
        "ファイル", [p.name for p in available],
        default=[p.name for p in available][:2],
    )
    for name in picked:
        path = DATA_DIR / name
        datasets.append(_load_from_path(str(path), path.stat().st_mtime))
elif source == "アップロード":
    uploads = st.sidebar.file_uploader(
        "ARW ファイル", type=["arw"], accept_multiple_files=True
    )
    for up in uploads or []:
        datasets.append(_load_from_bytes(up.name, up.getvalue()))
else:
    n = st.sidebar.slider("合成インジェクション数", 1, 4, 3)
    datasets = [synthetic_pdadata(injection_id=f"DEMO{i+1}") for i in range(n)]

st.sidebar.divider()
st.sidebar.subheader("検出トレース")
trace_kind = st.sidebar.radio(
    "ピーク検出に使うクロマトグラム",
    ["230 nm 単一波長", "MaxPlot(全波長最大)"], index=0,
    help="Empower の 230 nm 表示に合わせるには単一波長を選択。MaxPlot は全波長の最大吸光度の包絡線。",
)
if trace_kind.startswith("230"):
    base_wl = st.sidebar.number_input("検出波長 (nm)", 190.0, 800.0, 230.0, 1.0)
    base_bw = st.sidebar.number_input("バンド幅 (nm, 0=最近点)", 0.0, 20.0, 0.0, 1.0)
    prom_default, prom_max, prom_step = 0.0006, 0.02, 0.0001
else:
    base_wl, base_bw = None, 0.0
    prom_default, prom_max, prom_step = 0.02, 0.5, 0.005

st.sidebar.subheader("検出方法")
deconvolve = st.sidebar.checkbox(
    "重なり分離(デコンボリューション)", value=False,
    help="共溶出・ショルダーを成分フィットで分離し、被っても各ピークの RT・FWHM を取得。重回帰用。",
)
decon_model = "emg"
if deconvolve:
    decon_model = st.sidebar.selectbox(
        "ピークモデル", ["emg", "gauss"],
        format_func={"emg": "EMG(テーリング対応)", "gauss": "ガウス"}.get,
    )

st.sidebar.subheader("ピーク検出")
min_prominence = st.sidebar.slider(
    "最小プロミネンス (AU)", 0.0, prom_max, prom_default, prom_step, format="%.4f"
)
min_distance = st.sidebar.slider("最小ピーク間隔 (min)", 0.0, 1.0, 0.05, 0.01)
min_height_on = st.sidebar.checkbox("最小高さを使う", value=False)
min_height = st.sidebar.slider("最小高さ (AU)", 0.0, 1.0, 0.05, 0.01) if min_height_on else None

st.sidebar.subheader("ピーク高さ範囲フィルタ")
# default OFF: low-absorbance real peaks (e.g. late, high-retention runs at
# 230 nm) can fall below a height threshold — filtering risks losing them.
height_on = st.sidebar.checkbox("範囲外の高さのピークを除外", value=False)
height_unit = st.sidebar.radio(
    "高さの単位", ["AU", "µV"], index=0, horizontal=True,
    help="Empower と同じ AU、または検出器出力 µV(換算係数を指定)。",
)
if height_unit == "AU":
    uv_scale = 1.0
    height_min_au, height_max_au = st.sidebar.slider(
        "ピーク高さ範囲 (AU)", 0.0, 0.2, (0.001, 0.05), 0.001, format="%.3f",
        help="溶媒フロント(高)と微小ノイズ(低)を除外。230nm の解析ピークは概ね 0.0015–0.013 AU。",
    )
else:
    uv_scale = st.sidebar.number_input(
        "AU → µV 換算 (1 AU = ? µV)", 1.0, 10_000_000.0, 1_000_000.0, 1000.0,
        help="検出器の µV↔AU 換算係数(1 V/AU なら 1e6)。",
    )
    lo_uv, hi_uv = st.sidebar.slider(
        "ピーク高さ範囲 (µV)", 0, 200000, (1000, 50000), 500,
    )
    height_min_au, height_max_au = lo_uv / uv_scale, hi_uv / uv_scale

st.sidebar.subheader("モニタ波長フィルタ(任意)")
monitor_on = st.sidebar.checkbox("指定波長で吸収の弱いピークを除外", value=False)
monitor_wl = st.sidebar.number_input("モニタ波長 (nm)", 190.0, 800.0, 230.0, 1.0)
monitor_min_abs = st.sidebar.slider("除外閾値 A (AU)", 0.0, 0.2, 0.01, 0.001)

st.sidebar.subheader("表示 / トラッキング")
label_attr = st.sidebar.selectbox(
    "ピークラベル", ["apex_time", "lambda_max", "peak_id"],
    format_func={"lambda_max": "λmax", "peak_id": "ピークID", "apex_time": "保持時間 (RT)"}.get,
)
normalize = st.sidebar.checkbox("Y軸ノーマライズ", value=False)
track_spectral = st.sidebar.checkbox(
    "スペクトルで条件間トラッキング", value=True,
    help="UV スペクトル類似度で同定。条件を振って RT が大きくずれても同じ化合物を追跡。",
)
track_sequential = False
if track_spectral:
    min_sim = st.sidebar.slider("最小スペクトル類似度", 0.80, 1.0, 0.98, 0.01)
    track_sequential = st.sidebar.checkbox(
        "逐次連続性マッチング", value=True,
        help="各条件を直前条件に対して照合し、徐々にドリフトする RT を段階的に追跡。"
             "条件系列(グラジエント等)で安定。",
    )
    shift_label = ("隣接条件間の RT 最大シフト (min)" if track_sequential
                   else "RT 最大シフト (min)")
    rt_max_shift = st.sidebar.slider(shift_label, 0.3, 10.0,
                                     1.0 if track_sequential else 2.0, 0.1)
    rt_tol = 0.2
else:
    min_sim = 0.98
    rt_max_shift = None
    rt_tol = st.sidebar.slider("トラッキング RT 許容差 (min)", 0.01, 1.0, 0.2, 0.01)

wl_lo = float(min(d.wavelengths[0] for d in datasets)) if datasets else 190.0
wl_hi = float(max(d.wavelengths[-1] for d in datasets)) if datasets else 800.0
use_wl_range = st.sidebar.checkbox("MaxPlot 波長域を制限", value=False)
wl_range = None
if use_wl_range and datasets:
    wl_range = st.sidebar.slider(
        "MaxPlot 波長域 (nm)", wl_lo, wl_hi, (wl_lo, wl_hi)
    )

params = dict(
    base_wl=base_wl, base_bw=base_bw,
    deconvolve=deconvolve, decon_model=decon_model,
    min_prominence=min_prominence, min_distance=min_distance, min_height=min_height,
    monitor_on=monitor_on, monitor_wl=monitor_wl, monitor_min_abs=monitor_min_abs,
    height_on=height_on, uv_scale=uv_scale, height_unit=height_unit,
    height_min_au=height_min_au, height_max_au=height_max_au,
    wl_range=wl_range,
)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("PDA/UV 3D スペクトル解析")

if not datasets:
    st.info("← サイドバーでデータを選択してください(サンプル / アップロード / 合成デモ)。")
    st.stop()

results = [(d, *analyse(d, params)) for d in datasets]

# summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("インジェクション数", len(datasets))
c2.metric("検出ピーク合計", sum(len(t) for _, _, t, _, _ in results))
if monitor_on or height_on:
    dropped = sum(nb - len(t) for _, _, t, nb, _ in results)
    c3.metric("フィルタで除外", dropped)
c4.metric("波長点数", datasets[0].absorbance.shape[1])

tables = [t for _, _, t, _, _ in results]

st.header("① クロマトグラムと検出ピーク")
for data, mp, table, n_before, components in results:
    st.subheader(f"{data.metadata.injection_id} — {len(table)} ピーク"
                 + (f"(除外 {n_before - len(table)})" if (monitor_on or height_on) else ""))
    if components is not None:  # deconvolution view: separated components
        dt = float(np.median(np.diff(mp.times)))
        base = compute_baseline(mp.values, dt, DeconvolutionConfig(model=params["decon_model"]))
        chsub = Chromatogram(times=mp.times, values=mp.values - base,
                             label=mp.label, injection_id=mp.injection_id)
        st.pyplot(plot_deconvolution(chsub, components, table),
                  use_container_width=True)
        st.caption("黒=ベースライン減算した測定トレース、色=分離した各ピーク成分(重なりも分離)")
    else:
        fig = plot_labeled_chromatogram(
            mp, table, label_attr=label_attr, normalize=normalize,
            y_scale=uv_scale, y_unit=height_unit,
        )
        st.pyplot(fig, use_container_width=True)

    with st.expander("ピークテーブル / UV スペクトル / コンター"):
        df = peak_dataframe(table, monitor_wl, uv_scale, height_unit)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "CSV ダウンロード", df.to_csv(index=False).encode("utf-8"),
            file_name=f"peaks_{data.metadata.injection_id}.csv", mime="text/csv",
            key=f"dl_{data.metadata.injection_id}",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("各ピーク apex の UV スペクトル")
            st.pyplot(plot_uv_spectra(table), use_container_width=True)
        with col_b:
            st.caption("3D コンター (時間 × 波長、表示用に間引き)")
            st.pyplot(plot_contour(_decimate(data)), use_container_width=True)

if len(tables) > 1:
    st.header("② 条件間ピークトラッキング")
    result = track_peaks(tables, TrackingConfig(
        rt_tolerance=rt_tol, use_spectral=track_spectral,
        min_spectral_similarity=min_sim, rt_max_shift=rt_max_shift,
        sequential=track_sequential))
    matched = sum(1 for g in result.groups if len(g.members) == len(tables))
    st.caption(f"{len(result.groups)} グループ / 全条件で一致 {matched} "
               + ("(スペクトル類似度で同定)" if track_spectral else "(RT で同定)"))
    st.pyplot(plot_tracking(result), use_container_width=True)

    st.subheader("重回帰用テーブル(ピーク × 条件)")
    reg = regression_table(result)
    st.dataframe(reg, use_container_width=True, hide_index=True)
    col_r, col_f = st.columns(2)
    col_r.download_button(
        "重回帰テーブル (long) CSV", reg.to_csv(index=False).encode("utf-8"),
        file_name="regression_long.csv", mime="text/csv",
    )
    col_f.download_button(
        "FWHM 行列 CSV",
        result.to_dataframe(value="fwhm").to_csv(index=False).encode("utf-8"),
        file_name="tracking_fwhm.csv", mime="text/csv",
    )

st.caption(
    "検出トレース(230nm 等 / MaxPlot)でピーク検出(任意で重なり分離)→ apex スペクトルで λmax "
    "→ 任意フィルタ → 条件間トラッキング。単位: 時間=分, 波長=nm, 吸光度=AU。"
)
