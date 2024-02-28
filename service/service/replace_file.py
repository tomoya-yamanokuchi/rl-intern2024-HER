import shutil
from pathlib import Path


def replace_file(src: str, dest: str) -> None:
    src_path  = Path(src)
    dest_path = Path(dest)
    if src_path.exists() and src_path.is_file() and dest_path.exists() and dest_path.is_file():
        shutil.move(src_path, dest_path, copy_function=shutil.copy2)
        print(f"File {src} replaced {dest}")
    else:
        print(f"One of the files {src} or {dest} does not exist or is not a file")


if __name__ == '__main__':
    # 使用例
    src  = 'ディレクトリA/元のファイル名.txt'
    dest = 'ディレクトリB/置き換えるファイル名.txt'
    replace_file(src, dest)
