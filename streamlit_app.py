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
    PeakDetectionConfig,
    detect_peaks,
    filter_peaks_by_height,
)
from pda_peak_finder.plotting import (
    configure_japanese_font,
    plot_contour,
    plot_labeled_chromatogram,
    plot_tracking,
    plot_uv_spectra,
)
from pda_peak_finder.reader import load
from pda_peak_finder.spectra import (
    absorbance_at,
    annotate_peaks,
    filter_peaks_by_absorbance,
)
from pda_peak_finder.tracking import TrackingConfig, track_peaks
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
    mp = data.maxplot(wavelength_range=params["wl_range"])
    table = detect_peaks(
        mp,
        PeakDetectionConfig(
            min_height=params["min_height"],
            min_prominence=params["min_prominence"],
            min_distance_min=params["min_distance"],
        ),
    )
    table.source_label = mp.label
    annotate_peaks(data, table)
    n_before = len(table)
    if params["monitor_on"]:
        table = filter_peaks_by_absorbance(
            table, params["monitor_wl"], min_absorbance=params["monitor_min_abs"]
        )
    if params["height_on"]:
        # sidebar range is in µV; models are AU -> divide by the AU→µV scale
        table = filter_peaks_by_height(
            table,
            min_height=params["height_min_uv"] / params["uv_scale"],
            max_height=params["height_max_uv"] / params["uv_scale"],
        )
    return mp, table, n_before


def peak_dataframe(table, monitor_wl: float, uv_scale: float) -> pd.DataFrame:
    rows = []
    for p in table:
        a_mon = absorbance_at(p.spectrum, monitor_wl) if p.spectrum is not None else np.nan
        rows.append(
            {
                "peak_id": p.peak_id,
                "RT (min)": round(p.apex_time, 3),
                "FWHM (min)": round(p.fwhm, 4) if p.fwhm else None,
                "λmax (nm)": round(p.lambda_max, 1) if p.lambda_max else None,
                f"A@{monitor_wl:.0f}nm (AU)": round(float(a_mon), 5),
                "height (AU)": round(p.height, 4),
                "height (µV)": round(p.height * uv_scale, 1),
                "area (AU·min)": round(p.area, 5) if p.area else None,
            }
        )
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
st.sidebar.subheader("ピーク検出")
min_prominence = st.sidebar.slider("最小プロミネンス (AU)", 0.0, 0.5, 0.02, 0.005)
min_distance = st.sidebar.slider("最小ピーク間隔 (min)", 0.0, 1.0, 0.05, 0.01)
min_height_on = st.sidebar.checkbox("最小高さを使う", value=False)
min_height = st.sidebar.slider("最小高さ (AU)", 0.0, 1.0, 0.05, 0.01) if min_height_on else None

st.sidebar.subheader("モニタ波長フィルタ")
monitor_on = st.sidebar.checkbox("指定波長で吸収の弱いピークを除外", value=True)
monitor_wl = st.sidebar.number_input("モニタ波長 (nm)", 190.0, 800.0, 230.0, 1.0)
monitor_min_abs = st.sidebar.slider("除外閾値 A (AU)", 0.0, 0.2, 0.01, 0.001)

st.sidebar.subheader("ピーク高さ範囲フィルタ")
height_on = st.sidebar.checkbox("範囲外の高さのピークを除外", value=True)
uv_scale = st.sidebar.number_input(
    "AU → µV 換算 (1 AU = ? µV)", 1.0, 1_000_000.0, 1000.0, 100.0,
    help="このデータは AU 単位。ピーク高さを µV で扱うための換算係数。",
)
height_min_uv, height_max_uv = st.sidebar.slider(
    "ピーク高さ範囲 (µV)", 0, 20000, (100, 10000), 50
)

st.sidebar.subheader("表示 / トラッキング")
label_attr = st.sidebar.selectbox(
    "ピークラベル", ["lambda_max", "peak_id", "apex_time"],
    format_func={"lambda_max": "λmax", "peak_id": "ピークID", "apex_time": "保持時間"}.get,
)
normalize = st.sidebar.checkbox("Y軸ノーマライズ", value=False)
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
    min_prominence=min_prominence, min_distance=min_distance, min_height=min_height,
    monitor_on=monitor_on, monitor_wl=monitor_wl, monitor_min_abs=monitor_min_abs,
    height_on=height_on, uv_scale=uv_scale,
    height_min_uv=height_min_uv, height_max_uv=height_max_uv,
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
c2.metric("検出ピーク合計", sum(len(t) for _, _, t, _ in results))
if monitor_on or height_on:
    dropped = sum(nb - len(t) for _, _, t, nb in results)
    c3.metric("フィルタで除外", dropped)
c4.metric("波長点数", datasets[0].absorbance.shape[1])

tables = [t for _, _, t, _ in results]

st.header("① ラベル付きクロマトグラム (QDa/SIR スタイル)")
for data, mp, table, n_before in results:
    st.subheader(f"{data.metadata.injection_id} — {len(table)} ピーク"
                 + (f"(除外 {n_before - len(table)})" if monitor_on else ""))
    fig = plot_labeled_chromatogram(
        mp, table, label_attr=label_attr, normalize=normalize,
        y_scale=uv_scale, y_unit="µV",
    )
    st.pyplot(fig, use_container_width=True)

    with st.expander("ピークテーブル / UV スペクトル / コンター"):
        df = peak_dataframe(table, monitor_wl, uv_scale)
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
    st.header("② ピークトラッキング (分析間)")
    result = track_peaks(tables, TrackingConfig(rt_tolerance=rt_tol))
    st.pyplot(plot_tracking(result), use_container_width=True)
    st.caption("同一化合物と推定されるピークを RT で対応付け")
    matrix = result.to_dataframe(value="apex_time")
    st.dataframe(matrix, use_container_width=True, hide_index=True)
    st.download_button(
        "トラッキング RT 行列 CSV", matrix.to_csv(index=False).encode("utf-8"),
        file_name="tracking_rt.csv", mime="text/csv",
    )

st.caption(
    "MaxPlot(全波長の最大吸光度包絡線)でピーク検出 → apex スペクトルから λmax 算出 "
    f"→ {monitor_wl:.0f} nm 吸光度が閾値未満のピークを除外。単位: 時間=分, 波長=nm, 吸光度=AU。"
)
