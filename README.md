# 実行手順
## 環境構築
- Dockerコンテナを作成
```
docker compose build
```
- Dockerコンテナを起動
```
docker compose up -d
```
## 対話部分の学習/評価
### PPDPPディレクトリへ移動
```
cd PPDPP
```
### 言語モデルプラグインのファインチューニングによる初期化
- escデータセットを使用する場合
```
python sft.py --do_train --do_eval
```
- cbデータセットを使用する場合
```
python sft.py --data_name cb --do_train --do_eval
```
- cimaデータセットを使用する場合
```
python sft.py --data_name cima --do_train --do_eval
```
- 同じデータセットでファインチューニングを一度行っており、それに上書きしてもう一度はじめからファインチューニングをしたいとき(なにかトラブルなどが起きて不適切な履歴があるときなどに使用)  
データセットは使用したいものを上記のコマンドと同様に指定する。
```
python sft.py --do_train --do_eval --overwrite_output_dir
```
### .envファイルの作成
openAIのapikeyを取得して、PPDPPディレクトリの直下に.envファイルを以下の内容で作成し保存してください
```
OPENAI_API_KEY=(あなたのapikeyをここに書いてください)
```
### 言語モデルプラグインの強化学習
ファインチューニングを行ったデータセットに対して行ってください
- escデータセットを使用する場合
```
python run.py --do_train --do_eval
```
- cbデータセットを使用する場合
```
python run.py --data_name cb --do_train --do_eval
```
- cimaデータセットを使用する場合
```
python run.py --data_name cima --do_train --do_eval
```
## 対話の実践
### ディレクトリ移動
```
cd ..
```
### 対話の実行(音声サンプルファイルを入力とする)
```
python voice_chat.py
```