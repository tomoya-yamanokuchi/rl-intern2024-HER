import os


def find_files(directory: str, keyword: str):
    """
    Searches for files containing the keyword in their name within the given directory.

    :param directory: Directory to search in.
    :param keyword: Keyword to look for in the file names.
    :return: List of files that contain the keyword.
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if keyword in file:
                matching_files.append(os.path.join(root, file))
    return matching_files

# Example usage
# find_files('/path/to/directory', 'keyword')

