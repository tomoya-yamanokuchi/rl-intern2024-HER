from typing import List



def extract_numeric(labels: List[str], unit: str) -> List[int]:
    """
    ラベルリストから数値部分を抽出する関数。

    :param labels: 単位が含まれるラベルのリスト。
    :param unit: ラベルから取り除く単位の文字列。
    :return: 数値部分のみを含むリスト。
    """
    return [int(label.replace(unit, '')) for label in labels]
