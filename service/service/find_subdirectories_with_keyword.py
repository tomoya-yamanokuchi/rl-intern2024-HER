import os
from typing import List


def find_subdirectories_with_keyword(search_dir: str, keywords: List[str]):
    matching_subdirs = []
    if os.path.exists(search_dir):
        for subsubdir in os.listdir(search_dir):

            # スペースを除去する
            subsubdir_no_space = subsubdir.replace(" ", "")
            # for keyword in keywords:
                # # キーワードからもスペースを除去する
                # keyword_no_space = keyword.replace(" ", "")
            if all(keyword.replace(" ", "") in subsubdir_no_space for keyword in keywords):
            # if all(keyword in subsubdir for keyword in keywords):
                matching_subdirs.append(os.path.join(search_dir, subsubdir))
    return matching_subdirs



if __name__ == '__main__':
    # 例: ログディレクトリのリストと検索キーワードを設定
    log_dirs = [
        '/path/to/model1/logs',
        '/path/to/model2/logs',
        '/path/to/model3/logs'
    ]
    keyword = 'your_keyword'

    # 検索実行
    matched_subdirs = find_subdirectories_with_keyword(log_dirs, keyword)
    for subdir in matched_subdirs:
        print(subdir)
