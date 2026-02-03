from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -----------------------------
# App + Env
# -----------------------------
load_dotenv()
app = Flask(__name__)

# Debug check (optional)
print("GOOGLE_API_KEY loaded:", bool(os.getenv("GOOGLE_API_KEY")))

# -----------------------------
# Embeddings + FAISS
# -----------------------------
embeddings = download_hugging_face_embeddings()

FAISS_INDEX_PATH = "faiss_index"

db = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# Gemini LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    api_key=os.getenv("GOOGLE_API_KEY")
)


# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ]
)

# -----------------------------
# LCEL RAG Chain (LangChain 1.x)
# -----------------------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    answer = rag_chain.invoke(user_input)
    return answer


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
