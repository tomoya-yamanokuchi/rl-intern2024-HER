import shutil
from pathlib import Path


def file_copy_and_rename(src: str, dest: str, new_name: str) -> None:
    src_path  = Path(src)
    dest_path = Path(dest) / new_name
    if src_path.exists() and src_path.is_file():
        shutil.copy(src_path, dest_path)
        print(f"File {src} copied to {dest} with new name {new_name}")
    else:
        print(f"File {src} does not exist or is not a file")


if __name__ == '__main__':
    # 使用例
    src      = '/nfs/monorepo_ral2023/robel_dclaw_env/robel_dclaw_env/domain/environment/model/pushing_object/object_current.xml'
    dest     = '/nfs/monorepo_ral2023/env_dataset'

    new_name = 'koko.xml'
    file_copy_and_rename(src, dest, new_name)
