from typing import List
import numpy as np


def intra_inter_class_variance_ratio(x: np.ndarray, y: List[str]) -> float:
    """
    クラス内分散とクラス間分散の比を計算する関数。

    :param x: 特徴量の配列。
    :param y: 各特徴量に対応するクラスラベルのリスト。
    :return: クラス内分散とクラス間分散の比。
    """
    # クラスラベルごとにデータをグループ化
    unique_classes = set(y)
    class_data     = {cls: [] for cls in unique_classes}
    for value, label in zip(x, y):
        class_data[label].append(value)

    # クラス内分散の計算
    within_class_var = sum(np.var(class_data[cls]) for cls in unique_classes) / len(unique_classes)

    # クラス間分散の計算
    overall_mean      = np.mean(x)
    between_class_var = sum(len(class_data[cls]) * (np.mean(class_data[cls]) - overall_mean) ** 2 for cls in unique_classes) / len(x)

    # クラス内・クラス間分散比の計算
    var_ratio  = (between_class_var / within_class_var) if within_class_var != 0 else 0

    return var_ratio
