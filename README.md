# rl-intern2024-HER

## 使い方


1. `rl-intern2024-HER`のリポジトリをダウンロードする
2. ダウンロードした`rl-intern2024-HER`のディレクトリに移動する
3. 以下のコマンドで関連パッケージをインストールする

```
python3.8 -m pip install -r requirements.txt
sh package_install.sh
```

4. FetchPush環境でHERの学習が回ることを確認
```
mpirun -np 1 python3.8 -u ./usecase/her/train.py --env_name='FetchPush-v1'  --cuda | tee reach.log
```

5. 学習すると`saved_models`というディレクトリの直下に学習済みモデルが時刻名のディレクトリとして保存されていく
6. モデルが保存されたらargparseで`--env_name`（環境指定）と`--model`（どのモデルをロードするか）のパラメータを指定してdemo.pyを実行して学習済み方策の挙動を確認


```
# 例です
python3.8 -u ./usecase/her/demo.py --env_name='FetchPush-v1' --model='20240229_16_11_30'
```

___
## ROBEL DClaw サイコロ環境で動かす場合
`env_name`パラメータを`RobelDClawCube`と指定してtrain.pyとdemo.pyを上記の通り実行する

・train
```
mpirun -np 1 python3.8 -u ./usecase/her/train.py --env_name='RobelDClawCube'  --cuda | tee reach.log
```

・test
```
python3.8 -u ./usecase/her/demo.py --env_name='RobelDClawCube' --model='20240229_16_15_55 (これは例なので自分のロードするモデルのディレクトリ名に合わせてください)'
```
