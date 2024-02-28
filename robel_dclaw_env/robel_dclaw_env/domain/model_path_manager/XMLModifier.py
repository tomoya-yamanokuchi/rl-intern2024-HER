import os
import xml.etree.ElementTree as ET
import tempfile
import shutil


class XMLModifier:
    def modify_object_path(self, xml_path: str, new_file_path: str) -> str:
        tree         = ET.parse(xml_path)
        root         = tree.getroot()
        object1_body = root.find(".//body[@name='object1']")
        include_tag  = object1_body.find('.//include')
        include_tag.set('file', new_file_path)
        return ET.tostring(root, encoding='unicode')


    def modify_relative_paths_to_absolute(self, xml_string, base_path):
        lines = xml_string.split('\n')
        for i, line in enumerate(lines):
            if '<include file="' in line:
                relative_path = line.split('"')[1]
                # カレントディレクトリを表す './' を取り除いて結合
                absolute_path1 = os.path.join(base_path, relative_path.lstrip('./'))
                absolute_path2 = os.path.normpath(absolute_path1) # パスを正規化
                # import ipdb; ipdb.set_trace()
                lines[i] = line.replace(relative_path, absolute_path2)
        return '\n'.join(lines)



    def convert_relative_paths(self, xml_path, base_path=None):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 絶対パスのベースとなるパス
        if base_path is None:
            base_path = os.path.dirname(xml_path)

        # file属性を持つ全ての要素を取得
        for element in root.findall(".//*[@file]"):
            file_path = element.get('file')
            if file_path and file_path.startswith("./"):
                abs_path = os.path.join(base_path, file_path.lstrip("./"))
                element.set('file', abs_path)

        # include タグを探索して、それらが指すファイルも再帰的に処理
        for include in root.findall(".//include"):
            included_file_path = include.get('file')
            if included_file_path and included_file_path.startswith("./"):
                abs_included_path = os.path.join(base_path, included_file_path.lstrip("./"))
                include.set('file', abs_included_path)
                import ipdb; ipdb.set_trace()
                self.convert_relative_paths(abs_included_path, base_path)  # 再帰的に処理

        return ET.tostring(root, encoding="unicode")


    def copy_with_relative_paths(self, original_xml_path):
        # XMLファイルを解析
        tree = ET.parse(original_xml_path)
        root = tree.getroot()

        # 一時ディレクトリを作成
        temp_dir = tempfile.mkdtemp()

        # <include> タグに記述されたファイルをコピー
        original_dir = os.path.dirname(original_xml_path)
        for include_tag in root.findall(".//include"):
            file_path = include_tag.attrib['file']
            if not os.path.isabs(file_path):
                original_file_path = os.path.join(original_dir, file_path)
                new_relative_path = os.path.basename(file_path)
                new_path = os.path.join(temp_dir, new_relative_path)
                shutil.copy2(original_file_path, new_path)
                # includeタグのfile属性を更新
                include_tag.attrib['file'] = new_relative_path

        # オリジナルのXMLファイルを一時ディレクトリに保存
        new_xml_path = os.path.join(temp_dir, os.path.basename(original_xml_path))
        tree.write(new_xml_path)

        # 一時ファイルの内容を表示
        with open(new_xml_path, 'r') as file:
            print(file.read())

        return new_xml_path
