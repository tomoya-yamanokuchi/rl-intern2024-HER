from pathlib import Path


def get_files_with_suffix(search_dir: str, suffix: str):
    '''
    最初はosとglobモジュールを使って実装しようとしたが, []（カッコ）みたいな特殊文字が
    入っている場合には正常に解析できないため, pathlibでrglobを用いる方法に落ち着いた
    '''

    '''
    [search_dir配下にサブディレクトリがある場合でも対処可能か？]
    ・`Pathlib`の`rglob`メソッドはディレクトリの階層を再帰的に検索します。
    ・つまり、指定したディレクトリだけでなくそのサブディレクトリも含めて検索を行います。
    ・そのため、あなたが提供した`get_files_with_suffix`関数はすでに階層化されたディレクトリからも
      特定の接尾辞を持つファイルを検索できるはずです。
    ・具体的には、もし `search_dir` の直下に `sub_dir1` と `sub_dir2` があり、
      それらの中に目的のファイルがある場合でも、`search_dir` を `get_files_with_suffix` 関数の
      引数として与えれば、`sub_dir1` と `sub_dir2` の中のファイルも検索対象になります。
    '''

    # Create a Path object
    search_dir_path = Path(search_dir)

    # Use the rglob method, which can handle special characters
    files = list(search_dir_path.rglob('*.' + suffix))

    return files
