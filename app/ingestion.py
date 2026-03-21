import os 

def load_local_repo(path="data/"):
    documents = []
    allowed_extensions = (".py", ".js", ".ts", ".go", ".cpp", ".txt", ".md")
    
    for root, _, files in os.walk(path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in allowed_extensions:
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    documents.append({
                        "content": content,
                        "metadata": {
                            "file_path": file_path,
                            "file_name": file,
                            "extension": file_ext
                        }
                    })
                except Exception as e:
                    print(f"Ошибка при чтении {file_path}: {e}")
                    
    return documents