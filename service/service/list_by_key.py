from collections import defaultdict
from typing import List, Dict, Any


def list_by_key(dict_list: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    # defaultdictを使って、デフォルトの値を空のリストに設定
    result = defaultdict(list)

    # それぞれの辞書に対して
    for d in dict_list:
        # それぞれのキーと値に対して
        for key, pseud_value_object in d.items():
            # import ipdb; ipdb.set_trace()
            # 対応するリストに値を追加
            result[key].append(pseud_value_object.value)

    return result
