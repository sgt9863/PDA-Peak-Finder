# アーキテクチャ設計

PDA-Peak-Finder は Waters Empower から出力した PDA/UV の 3D スペクトルデータ
(時間 × 波長 × 吸光度)を解析するソフトウェアです。本ドキュメントは
**ARW フォーマットの詳細に依存しない**アーキテクチャを定義します。ファイル形式に
依存するのは `reader` モジュールのみで、それ以外はすべて共通データモデルの上で動作します。

## 設計原則

1. **データモデルを契約とする。** すべてのモジュールは `models.py` で定義した型を
   介してやり取りし、生ファイルには触れません。これによりファイル形式(ARW など)と
   解析ロジックが分離されます。
2. **リーダーは差し替え可能。** 新しい形式のサポートは `SpectralDataReader` を
   継承したクラスを 1 つ追加して登録するだけで済みます。他モジュールの変更は不要です。
3. **段階ごとに疎結合。** 検出・スペクトル・トラッキング・出力・可視化は独立モジュールで、
   `pipeline` が順序どおりに束ねます。各段は単体でもテスト・利用できます。
4. **単位を全体で固定。** 時間 = 分、波長 = nm、吸光度 = AU。リーダーが変換責任を持ちます。

## 想定ワークフロー

```
 1. データ読み込み      reader.load / load_many        -> PDAData
 2. MaxPlot 生成        PDAData.maxplot()              -> Chromatogram
 3. ピーク検出          peak_detection.detect_peaks    -> PeakTable
 4. ピーク特性計算       (RT / FWHM / area は 3 に内包)  -> PeakTable
 5. UV スペクトル抽出    spectra.annotate_peaks         -> PeakTable(λmax, spectrum 付与)
 6. ピークトラッキング   tracking.track_peaks           -> TrackingResult
 7. CSV 出力            export.write_*                 -> results/*.csv
 8. 可視化              plotting.plot_*                -> results/plots/*.png
```

`pipeline.run_pipeline()` がこの 1→8 を実行します。

## ディレクトリ構成

```
PDA-Peak-Finder/
├── pyproject.toml              # パッケージ定義・依存・CLI エントリポイント
├── README.md
├── docs/
│   ├── ARCHITECTURE.md         # 本ドキュメント
│   └── ROADMAP.md              # 開発ロードマップ
├── src/pda_peak_finder/
│   ├── __init__.py             # 公開 API(データモデル・例外)
│   ├── models.py               # ★ 共通データモデル(全モジュールの契約)
│   ├── errors.py               # 例外階層
│   ├── testing.py              # 合成データ生成(ARW 不要でテスト可能)
│   ├── pipeline.py             # ワークフローのオーケストレーション
│   ├── cli.py                  # CLI(analyze / demo)
│   ├── reader/                 # 入力: ファイル -> PDAData
│   │   ├── base.py             #   SpectralDataReader・レジストリ・load()
│   │   └── arw.py              #   Empower ARW リーダー(将来実装、現在はスタブ)
│   ├── peak_detection/         # ピーク検出と特性計算(RT/FWHM/area)
│   ├── spectra/                # UV スペクトル抽出・λmax 計算
│   ├── tracking/               # 分析間ピークトラッキング
│   ├── export/                 # CSV 出力
│   └── plotting/               # matplotlib 可視化
├── tests/                      # pytest(合成データで全段検証)
├── data/                       # 測定データ置き場(コミット対象外)
└── results/                    # 解析結果出力先(コミット対象外)
```

## データモデル(`models.py`)

全モジュールが共有する型。単位は 時間=分 / 波長=nm / 吸光度=AU で固定です。

| 型 | 役割 | 主なフィールド / メソッド |
|----|------|--------------------------|
| `InjectionMetadata` | 1 インジェクションの識別情報 | `injection_id`(下流の主キー), `sample_name`, `acquired_at`, `instrument`, `channel`, `source_path`, `extra` |
| `PDAData` | 1 インジェクションの 3D データ | `times` (T,), `wavelengths` (W,), `absorbance` (T,W), `metadata`。射影メソッド `maxplot()`, `chromatogram_at()`, `spectrum_at()` |
| `Chromatogram` | 時間 1 次元シグナル | `times`, `values`, `label`, `injection_id`, `sampling_interval` |
| `UVSpectrum` | ある RT の吸光度 vs 波長 | `wavelengths`, `values`, `time`, `label`, `injection_id` |
| `Peak` | 検出ピーク 1 個と算出値 | `apex_time`(RT), `apex_index`, `height`, `start_time`, `end_time`, `fwhm`, `area`, `lambda_max`, `spectrum`, `injection_id`, `peak_id`。`as_record()`, `EXPORT_COLUMNS` |
| `PeakTable` | 1 インジェクションのピーク集合 | `peaks`, `injection_id`, `source_label`。`to_dataframe()`, 反復可能 |
| `PeakGroup` | 分析間で同定した 1 化合物 | `group_id`, `members`(injection_id→Peak), `mean_rt`, `rt_std` |
| `TrackingResult` | 分析間のトラッキング結果 | `groups`, `injection_ids`。`to_dataframe(value=...)`(ワイド行列) |

**不変条件**(`__post_init__` で強制):`times`/`wavelengths` は狭義単調増加、
`absorbance` は `(T, W)`、`injection_id` は非空。これによりリーダーが不正データを
下流に渡すことは構造的に防止されます。

### データフロー概念図

```
ファイル ──reader──▶ PDAData ──maxplot──▶ Chromatogram
                       │                       │
                       │                  peak_detection
                       │                       ▼
                       │                   PeakTable ──spectra.annotate──▶ PeakTable(λmax, spectrum)
                       │                                                        │
                       ▼                                                   tracking(複数)
                  spectrum_at                                                   ▼
                   (λmax 抽出)                                            TrackingResult
                                                                    │            │
                                                              export ◀──────────┘
                                                              plotting
```

## 各モジュールの責務

### reader — 入力(ファイル → PDAData)
- ファイル 1 つを検証済み `PDAData` に変換する唯一の場所。
- `SpectralDataReader`(抽象基底):`format_name`, `file_patterns`, `sniff()`, `read()`。
- レジストリ(`register_reader` / `get_reader` / `load` / `load_many`)がフォーマット判定と
  読み込みの入口。フォーマット追加は 1 クラス追加+登録のみ。
- `arw.py` は Waters Empower ARW リーダー(実装済み)。Shift-JIS(cp932)・CR 改行・
  タブ区切りで、`波長`行(波長軸)+ `時間`ブロック(各行 = 時間 + 各波長の吸光度)を
  構造ベースに解析(日本語ラベル非依存)。truncated 末尾行はスキップ、降順波長はソート。

### peak_detection — ピーク検出と特性計算
- `Chromatogram`(通常は MaxPlot)から全ピークを検出し、`PeakTable` を返す。
- 各ピークに RT(apex_time)・height・積分区間(start/end)・**FWHM**・area を付与。
- 実装は `scipy.signal`(`find_peaks`, `peak_widths`, 任意で `savgol_filter`)。
- 調整パラメータは `PeakDetectionConfig`(prominence・最小間隔・平滑化など)。

### spectra — UV スペクトル抽出・λmax
- `PDAData` から各ピーク apex の UV スペクトルを抽出。
- **λmax**(吸光度最大の波長)を算出し、`PeakTable` の各 `Peak` に `spectrum`/`lambda_max` を付与。
- apex 周辺スキャン平均・波長方向平滑化・ベースライン減算を `SpectrumConfig` で制御。

### tracking — 分析間ピークトラッキング
- 複数 `PeakTable` を受け取り、同一化合物と推定されるピークを RT(任意で λmax)で対応付け。
- `TrackingResult`(`PeakGroup` の集合)を返す。決定的な貪欲マッチング。
- 許容差などは `TrackingConfig`。

### export — CSV 出力
- `PeakTable` / `TrackingResult` / UV スペクトルを CSV 化(pandas)。
- 親ディレクトリ自動作成、書き込んだ `Path` を返す。

### plotting — 可視化
- matplotlib(Agg バックエンド)で図を生成し `Figure` を返す。
- クロマトグラム+ピーク注釈、3D コンター、UV スペクトル重ね描き、トラッキング図、保存ヘルパー。

### pipeline / cli — オーケストレーションと CLI
- `pipeline` がワークフロー 1→8 を束ね、`AnalysisConfig` を保持。
- `cli` は `analyze`(ファイル解析)と `demo`(合成データ)の 2 サブコマンド。

## リーダーインターフェース設計

ARW リーダーは**実装済み**です(`reader/arw.py`)。同じ契約に沿って、新しいフォーマットも
`read()` を実装するだけで全体が動きます。契約は以下のとおりです。

```python
class SpectralDataReader(ABC):
    format_name: ClassVar[str]                 # 例: "arw"
    file_patterns: ClassVar[tuple[str, ...]]   # 例: ("*.arw",)

    @classmethod
    def sniff(cls, path: Path) -> bool: ...     # 安価な判定(既定はファイル名パターン)

    @abstractmethod
    def read(self, path: Path) -> PDAData: ...  # 本体: 検証済み PDAData を返す
```

`read()` の契約:
- 返す `PDAData` は 分 / nm / AU、両軸は狭義単調増加(必要ならソート/反転)。
- `metadata.injection_id` を必ず設定(ヘッダに無ければファイル名 stem を使う)。
- 認識できない場合は `ReaderError`(可能なら行番号付き)。
- `PDAData.__post_init__` が形状・単調性・ID を検証するため、通常の構築フローで
  不正データが下流に流れることはない。

新フォーマット追加手順:`SpectralDataReader` を継承 → `read`/`sniff` を実装 →
`reader/__init__.py` で `register_reader()`。呼び出し側(`pipeline`/`cli`)は無変更。

ARW の具体的なフォーマット仕様と実装上の判断は `reader/arw.py` の docstring を参照。

## テスト戦略

`testing.synthetic_pdadata()` が既知の RT・FWHM・λmax を持つ合成 `PDAData` を生成するため、
**ARW ファイルなしで**検出・スペクトル・トラッキング・出力・可視化の全段を
グラウンドトゥルースに対して検証できます。ARW リーダーは、テスト内で同フォーマットの
小さな ARW を生成するラウンドトリップテスト(`tests/test_arw.py`)で検証し、
`data/` に実サンプルがある場合はそれも読み込んで検証します(存在しなければスキップ)。
