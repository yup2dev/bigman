import os

def print_directory_structure(start_path='.', indent=''):
    for entry in os.listdir(start_path):
        path = os.path.join(start_path, entry)
        if os.path.isdir(path):
            print(f"{indent}📁 {entry}/")
            print_directory_structure(path, indent + '    ')
        else:
            print(f"{indent}📄 {entry}")

# 사용 예시
print("📂 Current Directory Structure:")
print_directory_structure()