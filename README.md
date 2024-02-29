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
mpirun -np 1 python3.8 -u ./usecase/her/train.py --env-name='FetchPush-v1'  --cuda | tee reach.log
```

5. 学習すると`saved_models`というディレクトリの直下に学習済みモデルが保存されていく
6. モデルが保存されたらdemo.pyを実行して学習済み方策の挙動を確認
