import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Load Groq API key from secrets (hidden from visitors)
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("⚠️ Groq API key not configured. Please contact the site owner.")
    st.stop()

# Main title
st.title("Book of Revelation - Chapter 1 AI Assistant")
st.markdown("Ask any question about **Chapter 1** of my Book of Revelation. The AI will answer using **only** the text from Chapter 1.")

# Automatically load your Chapter 1 file
@st.cache_resource
def load_chapter():
    try:
        loader = TextLoader("data/chapter1.txt", encoding="utf-8")
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading chapter: {e}")
        return []

documents = load_chapter()

if not documents:
    st.stop()

# Split the text into small pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

st.success("✅ Chapter 1 loaded successfully. You can now ask questions!")

# === AI CHATBOT SETUP FOR CHAPTER 1 ===

if "messages" not in st.session_state:
    st.session_state.messages = []

# Use Groq (fast & free tier)
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # Larger context window (128k tokens)
    temperature=0.3,
    groq_api_key=groq_api_key
)

# Create vector store from your Chapter 1
@st.cache_resource
def create_vectorstore():
    # Use a free local embedding model (no API key needed)
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

vectorstore = create_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Strong instruction so the bot stays ONLY in your Chapter 1
# Strong system prompt - must include {context} for stuff chain
system_prompt = """You are a helpful assistant for Chapter 1 of the Book of Revelation.

Context from Chapter 1:
{context}

Answer the question using ONLY the above context.
If the answer is not in the context, reply exactly: "This is not mentioned in Chapter 1."
Keep answers short and direct."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
if user_input := st.chat_input("Ask any question about Chapter 1..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.messages[:-1]
            })
            answer = response["answer"]
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
