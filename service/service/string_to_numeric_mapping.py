from .extract_radius import extract_radius
import matplotlib.cm as cm
import numpy as np


def extract_radius_from_batch_string_list(string_list: list):
    return [extract_radius(name) for name in string_list]

def string_to_numeric_mapping(string_list, verbose=False):
    labels            = extract_radius_from_batch_string_list(string_list)            # 円柱の半径情報を抽出
    unique_labels     = list(set(labels))                                             # 重複を削除したユニークな文字列の集合
    num_unique_labels = len(unique_labels)                                            # ユニークなラベルの数を取得
    colors            = cm.rainbow(np.linspace(0, 1, num_unique_labels))              # カラーマップの選択とカラーの生成
    color_map         = {label: color for label, color in zip(unique_labels, colors)} # カラーマッピングの生成
    # ---
    return labels, color_map


if __name__ == '__main__':
    # 1次元データとラベルの生成
    data   = np.random.randn(50)
    labels = np.random.choice(['A', 'B', 'C'], size=50)
