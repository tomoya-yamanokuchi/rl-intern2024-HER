import random
import string


def generate_random_string(length: int = 7):
    letters = string.ascii_lowercase  # 小文字のアルファベットだけを使用
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str


if __name__ == '__main__':
    # 使用例
    random_string = generate_random_string(7)  # 10文字のランダムな文字列を生成
    print("Random string:", random_string)
