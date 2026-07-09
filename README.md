# PDA-Peak-Finder

Waters Empower から取得した **PDA/UV の 3D スペクトルデータ**(時間 × 波長 × 吸光度)を
解析する Python ソフトウェアです。UV 吸収を持つ全ピークを自動検出し、各ピークの
Retention Time (RT)・FWHM・λmax を算出、複数分析間でピークをトラッキングし、
CSV 出力と可視化を行います。

> **現在のステータス:** アーキテクチャ・全解析パイプライン・**Waters Empower ARW リーダー**を実装済み。
> `pda-peaks analyze data/*.arw` で実データを解析できます。ファイルがなくても組み込みの
> 合成データ(`pda-peaks demo`)で全機能を試せます。

## 特長

- 3D データからの自動ピーク検出(`scipy.signal` ベース)
- **重なり分離(デコンボリューション)**:共溶出・ショルダーを EMG/ガウス成分にフィットし、
  被っても各ピークの RT・FWHM を分離抽出(重回帰用データ向け)
- 検出トレースは **230 nm 等の単一波長**(Empower の 230nm 表示に一致)または MaxPlot を選択
- ピークごとの RT / FWHM / 面積 / **λmax** と UV スペクトル抽出
- **モニタ波長フィルタ**:指定波長(例 230 nm)で吸収の弱いピークを除外
- 複数インジェクション間のピークトラッキング(RT ベース、任意で λmax 併用)
- CSV 出力(ピーク表・トラッキング行列・スペクトル)
- 可視化:**QDa/SIR 風のラベル付きクロマトグラム**(λmax ラベル・任意で Y 軸ノーマライズ)、
  3D コンター・UV スペクトル・トラッキング図
- **Streamlit Web アプリ**(`streamlit run streamlit_app.py`)
- numpy / pandas / scipy / matplotlib のみ(アプリは streamlit を追加)

## インストール

```bash
python -m pip install -e ".[dev]"   # 開発用(pytest 含む)
# または
python -m pip install -e .
```

Python 3.10 以上が必要です。

## クイックスタート

ファイルがなくても、組み込みの合成データで全パイプラインを実行できます:

```bash
pda-peaks demo -o results/
```

実データ(Waters Empower ARW エクスポート)を解析する場合:

```bash
pda-peaks analyze data/*.arw -o results/
```

ARW は Shift-JIS・CR 改行・タブ区切りの 3D エクスポート(時間 × 波長)に対応しています。
低波長カットオフ(溶媒吸収)で λmax が下限に張り付く場合は
`--wavelength-range 210 400` などで解析範囲を絞れます。

主なオプション:`--min-prominence`(検出感度)、`--min-distance`(最小ピーク間隔・分)、
`--monitor-wavelength 230 --monitor-min-abs 0.01`(230 nm で吸収の弱いピークを除外)、
`--rt-tolerance`(トラッキング許容差・分)、`--wavelength-range LO HI`(MaxPlot の波長範囲)、
`--no-tracking` / `--no-plots`。詳細は `pda-peaks analyze -h`。

### Streamlit Web アプリ

```bash
python -m pip install -e ".[app]"     # streamlit を追加インストール
streamlit run streamlit_app.py
```

ブラウザ上で ARW をアップロード(または `data/` のサンプル・合成デモを選択)し、検出パラメータ・
モニタ波長・除外閾値・ラベル種別・Y 軸ノーマライズをスライダーで調整しながら、ラベル付き
クロマトグラム/ピークテーブル/UV スペクトル/コンター/トラッキングを表示し CSV を出力できます。

Python API:

```python
from pda_peak_finder.pipeline import run_pipeline, AnalysisConfig

result = run_pipeline(["data/a.arw", "data/b.arw"], output_dir="results/")
for table in result.tables:
    print(table.injection_id, len(table), "peaks")
    print(table.to_dataframe())
print(result.tracking.to_dataframe(value="apex_time"))
```

## 解析ワークフロー

```
1. データ読み込み    reader.load / load_many              -> PDAData
2. MaxPlot 生成      PDAData.maxplot()                    -> Chromatogram
3. ピーク検出        peak_detection.detect_peaks          -> PeakTable
4. ピーク特性計算     (RT / FWHM / area は 3 に内包)
5. UV スペクトル抽出  spectra.annotate_peaks               -> λmax / spectrum 付与
5b. 波長フィルタ      spectra.filter_peaks_by_absorbance   -> 230 nm 等で除外(任意)
6. ピークトラッキング tracking.track_peaks                 -> TrackingResult
7. CSV 出力          export.write_*
8. 可視化            plotting.plot_*(QDa/SIR 風含む)
```

`pipeline.run_pipeline()` がこの一連を実行します。

## プロジェクト構成

```
src/pda_peak_finder/
  models.py         共通データモデル(全モジュールの契約)
  reader/           ファイル -> PDAData(ARW リーダーは差し替え可能)
  peak_detection/   ピーク検出・RT/FWHM/面積
  spectra/          UV スペクトル抽出・λmax
  tracking/         分析間ピークトラッキング
  export/           CSV 出力
  plotting/         matplotlib 可視化
  pipeline.py       ワークフローのオーケストレーション
  cli.py            CLI(analyze / demo)
  testing.py        合成データ生成(ARW 不要でテスト可能)
tests/              pytest(合成データで全段検証)
data/               測定データ置き場(コミット対象外)
results/            解析結果出力先(コミット対象外)
```

詳細な設計(データモデル・各モジュールの責務・リーダーインターフェース)は
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)、開発計画は
[docs/ROADMAP.md](docs/ROADMAP.md) を参照してください。

## 開発

```bash
python -m pytest              # 全テスト
python -m pytest tests/test_peak_detection.py -q   # 単一モジュール
python -m pytest tests/test_peak_detection.py::test_name   # 単一テスト
```

## 依存ライブラリ

numpy, pandas, scipy, matplotlib(実行時)/ pytest(開発時)。
