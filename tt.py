import os

def print_directory_structure(start_path='.', indent='', max_depth=2, current_depth=0):
    if current_depth >= max_depth:
        return
    for entry in os.listdir(start_path):
        path = os.path.join(start_path, entry)
        if os.path.isdir(path):
            print(f"{indent}📁 {entry}/")
            print_directory_structure(path, indent + '    ', max_depth, current_depth + 1)
        else:
            print(f"{indent}📄 {entry}")

# 예시 사용
print("📂 Directory Structure (max depth = 3):")
print_directory_structure('.', max_depth=3)
