# prediction_racing
競馬成績を予想するAIモデル

2001年～2021年の競馬データ（成績、騎手、生産者、産地、馬など）を加工処理、
設けたデータ構造でAIモデルトレニンーグを行った。
pyTorchのresnet50は標準モデルとする。
認証の際に、2021年の一部のデータで支払金の104%を還元できることを観測した。
データ・モデル詳細はarrange_data.ipynbご参照ください。
 
### 技術スタック
Python、Flask、pyTorch、pandas、PostgreSQL