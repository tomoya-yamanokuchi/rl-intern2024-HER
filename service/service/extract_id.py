import re

def extract_id(text):
    match = re.search(r'id=(\d+)', text)
    if match:
        return match.group(1)
    else:
        return None

if __name__ == '__main__':
    # テスト
    print(extract_id("[id=20230909132920]-[model_class=motion_dependent_content_cdsvae]"))  # 出力: '20230909132920'
    print(extract_id("[id=20230909132610]-[model_class=motion_dependent_content_cdsvae]"))  # 出力: '20230909132610'
