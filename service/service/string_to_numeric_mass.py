from .extract_radius import extract_radius
import matplotlib.cm as cm
import numpy as np


def extract_radius_from_batch_string_list(string_list: list):
    return [name.split("_mass_")[-1] for name in string_list]


# def convert_num_to_symbol(labels):
#     symbol_list = []
#     for _label in labels:
#         if _label == "1e_01_1e_02_1e_02_1e_02": new_label = "hlll"
#         if _label == "1e_01_1e_01_1e_02_1e_02": new_label = "hhll"
#         if _label == "1e_01_1e_01_1e_01_1e_02": new_label = "hhhl"
#         if _label == "1e_01_1e_02_1e_01_1e_02": new_label = "hlhl"
#         if _label == "1e_01_1e_01_1e_02_1e_01": new_label = "hhlh"

#         if _label == "1e_01_1e_01_1e_01_1e_01": new_label = "hhhh"
#         if _label == "1e_02_1e_01_1e_01_1e_02": new_label = "lhhl"
#         if _label == "1e_01_1e_02_1e_02_1e_01": new_label = "hllh"
#         if _label == "1e_02_1e_02_1e_02_1e_02": new_label = "llll"

#         if _label == "1e_02_1e_01_1e_01_1e_01": new_label = "lhhh"
#         if _label == "1e_02_1e_02_1e_01_1e_01": new_label = "llhh"
#         if _label == "1e_02_1e_02_1e_02_1e_01": new_label = "lllh"
#         if _label == "1e_01_1e_02_1e_01_1e_01": new_label = "hlhh"
#         if _label == "1e_02_1e_01_1e_02_1e_01": new_label = "lhlh"
#         symbol_list.append(new_label)
#     return symbol_list


# def convert_num_to_symbol(labels):
#     symbol_list = []
#     for _label in labels:
#         # ------ trajectory level Left COG --------
#         if _label == "1p0e01_1p0e02_1p0e02_1p0e02": new_label = "left-1"
#         if _label == "1p0e01_1p0e02_1p0e01_1p0e02": new_label = "left-2"
#         if _label == "4p0e02_2p0e01_1p0e01_1p0e02": new_label = "left-3"
#         if _label == "5p0e02_2p0e01_1p0e01_4p0e02": new_label = "left-4"
#         if _label == "1p0e02_2p0e01_1p5e01_1p0e02": new_label = "left-5"
#         # ------ trajectory level Center COG ------
#         if _label == "1p0e01_1p0e01_1p0e01_1p0e01": new_label = "center-6"
#         if _label == "1p0e02_1p0e01_1p0e01_1p0e02": new_label = "center-7"
#         if _label == "1p0e02_1p0e02_1p0e02_1p0e02": new_label = "center-8"
#         # ------ trajectory level Right COG ------
#         if _label == "5p6e02_1p0e01_1p0e02_1p0e01": new_label = "right-9"
#         if _label == "5p0e02_1p0e01_1p0e02_1p0e01": new_label = "right-10"
#         if _label == "4p0e02_1p0e01_2p0e01_5p0e02": new_label = "right-11"
#         if _label == "3p7e02_1p0e01_1p0e02_1p0e01": new_label = "right-12"
#         if _label == "1p0e02_1p0e01_2p0e01_4p0e02": new_label = "right-13"
#         if _label == "1p3e02_1p0e01_1p0e02_1p0e01": new_label = "right-14"
#         if _label == "1p0e02_1p0e02_1p0e02_1p0e01": new_label = "right-15"
#         symbol_list.append(new_label)
#     return symbol_list


def convert_num_to_symbol(labels):
    symbol_list = []
    for _label in labels:
        # ------ content level Left COG --------
        if _label == "1p0e01_1p0e02_1p0e02_1p0e02": new_label = "left-1"
        if _label == "1p0e01_1p0e02_1p0e01_1p0e02": new_label = "left-2"
        if _label == "4p0e02_2p0e01_1p0e01_1p0e02": new_label = "left-3"
        # ------ content level Center COG ------
        if _label == "1p0e01_1p0e01_1p0e01_1p0e01": new_label = "center-4"
        # ------ content level Right COG ------
        if _label == "1p0e02_1p0e01_2p0e01_4p0e02": new_label = "right-5"
        if _label == "1p0e02_1p0e01_1p0e02_1p0e01": new_label = "right-6"
        if _label == "1p0e02_1p0e02_1p0e02_1p0e01": new_label = "right-7"
        symbol_list.append(new_label)
    return symbol_list


def string_to_numeric_mass(string_list, verbose=False):
    labels            = extract_radius_from_batch_string_list(string_list)            # 円柱の半径情報を抽出
    # import ipdb; ipdb.set_trace()
    labels            = convert_num_to_symbol(labels)
    unique_labels     = list(set(labels))                                             # 重複を削除したユニークな文字列の集合
    num_unique_labels = len(unique_labels)                                            # ユニークなラベルの数を取得
    colors            = cm.rainbow(np.linspace(0, 1, num_unique_labels))              # カラーマップの選択とカラーの生成
    color_map         = {label: color for label, color in zip(unique_labels, colors)} # カラーマッピングの生成
    # ---
    # import ipdb; ipdb.set_trace()
    return labels, color_map


if __name__ == '__main__':
    # 1次元データとラベルの生成
    data   = np.random.randn(50)
    labels = np.random.choice(['A', 'B', 'C'], size=50)
