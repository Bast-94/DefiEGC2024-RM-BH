import os
def find_directory(root_directory, target_directory):
    for root, dirs, files in os.walk(root_directory):
        if target_directory in dirs:
            return os.path.join(root, target_directory)
    return None