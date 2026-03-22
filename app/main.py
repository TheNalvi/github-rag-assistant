from ingestion import load_local_repo
from dotenv import load_dotenv
from chunking import get_code_splitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

docs = load_local_repo("../data/")
all_chunks = []
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

for doc in docs:
    content = doc["content"]
    metadata = doc["metadata"]
    file_ext = metadata["extension"]
    
    splitter = get_code_splitter(file_ext, chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(content)
    
    for chunk in chunks:
        all_chunks.append(Document(page_content=chunk, metadata=metadata))

vector_db = FAISS.from_documents(all_chunks, embeddings)
vector_db.save_local("faiss_index")

api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

template = """
You are a professional software engineer assistant. 
Use the following pieces of retrieved code to answer the user's question.
If the answer is not in the context, say that you don't know. 
Be concise and explain the code if necessary.

CONTEXT:
{context}

QUESTION: 
{question}

ANSWER:
"""
prompt = ChatPromptTemplate.from_template(template)

def ask_gemini(query, vector_db):
    docs = vector_db.similarity_search(query, k=3)
    context = "\n---\n".join([d.page_content for d in docs])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({"context": context, "question": query})

user_query = "How is the user login handled and what security is used?"
response = ask_gemini(user_query, vector_db)

print("\n🤖 Gemini's Response:")
print(response)