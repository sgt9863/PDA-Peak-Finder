# results/

解析結果(CSV・プロット)の出力先です。

- 出力ファイルはリポジトリにコミットしません(`.gitignore` で除外)。
- `pda-peaks analyze ... -o results/` で以下が生成されます:
  - `peaks_all.csv` … 全インジェクションのピーク一覧(long 形式)
  - `peaks_<injection>.csv` … インジェクションごとのピーク表
  - `spectra_<injection>.csv` … 各ピークの UV スペクトル(long 形式)
  - `tracking_rt.csv` / `tracking_lambda_max.csv` … トラッキング行列
  - `plots/` … クロマトグラム・コンター・UV スペクトル・トラッキング図
