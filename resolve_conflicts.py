import os

def resolve_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    out = []
    for line in lines:
        if line.startswith('<<<<<<<'):
            continue
        if line.startswith('======='):
            continue
        if line.startswith('>>>>>>>'):
            continue
        out.append(line)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(out)

resolve_file('src/aitutor/api/main.py')
resolve_file('src/aitutor/api/static/index.html')
resolve_file('src/aitutor_ncert_rag.egg-info/PKG-INFO')
resolve_file('src/aitutor_ncert_rag.egg-info/SOURCES.txt')

print("Conflicts resolved.")
