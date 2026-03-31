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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Set page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Sidebar for API key and instructions
st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Groq API Key (free at groq.com)", type="password")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
else:
    st.sidebar.warning("Please enter your Groq API Key to start chatting.")
    st.stop()

st.sidebar.markdown("""
- **General Chat**: Start chatting immediately with the AI.
- **RAG Chat**: Upload files (PDF, TXT, CSV, DOCX, etc.) and index them to query your data.
- **Note**: For PDFs, ensure Poppler is installed and in PATH. Alternatively, use TXT or DOCX files.
- **Troubleshooting**: If answers are "I don't know," check if files loaded correctly or try more specific questions.
""")

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
    model="llama3-8b-8192",
    temperature=0.3,
    groq_api_key=groq_api_key
)

# Create vector store from your Chapter 1
@st.cache_resource
def create_vectorstore():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

vectorstore = create_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Strong instruction so the bot stays ONLY in your Chapter 1
system_prompt = """You are a helpful assistant for Chapter 1 of my Book of Revelation.
Answer EVERY question using ONLY the information from the provided Chapter 1 text.
If the answer is not mentioned in Chapter 1, simply reply: "This is not mentioned in Chapter 1."
Do not add any outside knowledge or information from other chapters."""

# Build the RAG chain
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
