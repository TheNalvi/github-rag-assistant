import os
import shutil
import stat
import logging
import git
import gradio as gr
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from ingestion import load_local_repo
from chunking import get_code_splitter

#CONFIG
logging.basicConfig(level=logging.INFO)
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_PATH = "faiss_index"
REPO_PATH = "./temp_repo"
MAX_FILES = 2000

#HELPERS
def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clean_repo():
    if os.path.exists(REPO_PATH):
        shutil.rmtree(REPO_PATH, onerror=remove_readonly)

#MODELS
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

vector_db = None

#INDEXING
def index_repository(repo_url):
    global vector_db

    if not repo_url.startswith("https://github.com/"):
        return "❌ Only GitHub repositories are supported"

    try:
        clean_repo()

        logging.info(f"Cloning {repo_url}")
        git.Repo.clone_from(repo_url, REPO_PATH, depth=1)

        docs = load_local_repo(REPO_PATH)
        docs = docs[:MAX_FILES]

        all_chunks = []

        for doc in docs:
            splitter = get_code_splitter(doc["metadata"]["extension"])
            chunks = splitter.split_text(doc["content"])

            for chunk in chunks:
                all_chunks.append(
                    Document(
                        page_content=chunk,
                        metadata=doc["metadata"]
                    )
                )

        vector_db = FAISS.from_documents(all_chunks, embeddings)
        vector_db.save_local(DB_PATH)

        return "✅ Repository indexed and saved!"

    except Exception as e:
        logging.exception("Indexing failed")
        return f"❌ Error: {str(e)}"

    finally:
        clean_repo()

#CHAT 
def load_db():
    global vector_db

    if vector_db is None and os.path.exists(DB_PATH):
        vector_db = FAISS.load_local(DB_PATH, embeddings)

    return vector_db

def format_history(history):
    formatted = []

    for msg in history:
        if msg["role"] == "user":
            formatted.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            formatted.append(f"Assistant: {msg['content']}")

    return "\n".join(formatted)

def build_context(docs):
    context_parts = []

    for d in docs:
        context_parts.append(
            f"FILE: {d.metadata.get('file_path', 'unknown')}\n"
            f"----------------\n"
            f"{d.page_content}"
        )

    return "\n\n".join(context_parts[:5])

def chat_fn(message, history):
    db = load_db()

    if db is None:
        return "Please index a repository first"

    history = history[-10:]
    chat_history = format_history(history)

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20}
    )

    docs = retriever.invoke(message)
    context = build_context(docs)

    template = """
    You are a senior software engineer analyzing a codebase.

    Conversation history:
    {history}

    Rules:
    - Answer ONLY using the provided context
    - If unsure → say "I don't know"
    - Be concise and technical
    - Understand references like 'it', 'this', 'they'
    - Reference file names when possible

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": message,
        "history": chat_history
    })

#UI
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🤖 GitHub RAG Assistant")
    gr.Markdown("Index a repository and ask questions about the code.")

    with gr.Row():
        repo_input = gr.Textbox(
            label="GitHub URL",
            placeholder="https://github.com/user/repo",
            scale=4
        )
        index_btn = gr.Button("Index", variant="primary")

    status = gr.Markdown("Status: Ready")

    chatbot = gr.ChatInterface(
        fn=chat_fn,
        examples=[
            "Explain the architecture",
            "Where is the main logic?",
            "How does authentication work?"
        ],
        cache_examples=False
    )

    index_btn.click(
        fn=index_repository,
        inputs=[repo_input],
        outputs=[status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)