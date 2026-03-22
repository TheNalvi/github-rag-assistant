import os
import shutil
import git
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ingestion import load_local_repo
from chunking import get_code_splitter
from langchain_core.documents import Document
import stat

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)
    
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

current_db = None
def index_repository(repo_url):
    global current_db
    repo_path = "./temp_repo"
    
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=remove_readonly)
    
    print(f"Cloning {repo_url}...")
    git.Repo.clone_from(repo_url, repo_path, depth=1)
    
    docs = load_local_repo(repo_path)
    all_chunks = []
    for doc in docs:
        splitter = get_code_splitter(doc["metadata"]["extension"])
        chunks = splitter.split_text(doc["content"])
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk, metadata=doc["metadata"]))
    
    current_db = FAISS.from_documents(all_chunks, embeddings)
    return "✅ Репозиторий проиндексирован! Можешь задавать вопросы."

def chat_fn(message, history):
    if current_db is None:
        return "Сначала вставь ссылку на GitHub и нажми 'Index'."
    
    docs = current_db.similarity_search(message, k=3)
    context = "\n---\n".join([d.page_content for d in docs])
    
    template = "Answer based on code:\n{context}\nQuestion: {question}"
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({"context": context, "question": message})

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🤖 GitHub Assistant")
    
    with gr.Row():
        repo_input = gr.Textbox(label="GitHub Repo URL", placeholder="https://github.com/user/repo")
        index_btn = gr.Button("Index Repository")
    
    status_msg = gr.Markdown("Status: Waiting for repository...")
    
    chatbot = gr.ChatInterface(fn=chat_fn)
    
    index_btn.click(index_repository, inputs=[repo_input], outputs=[status_msg])

if __name__ == "__main__":
    demo.launch()