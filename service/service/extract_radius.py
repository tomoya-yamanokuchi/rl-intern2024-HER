import re

def extract_radius(text):
    match = re.search(r'radius_(\d+mm)', text)
    if match:
        return match.group(1)
    else:
        return None
