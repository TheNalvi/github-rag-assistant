import os 

def load_local_repo(path="data/"):
    documents = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith((".py", ".js", ".txt", ".md")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
    return documents