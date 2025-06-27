import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# ✅ Correct import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
#from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
#from langchain.memory import ConversationSummaryBufferMemory
import shutil
from sentence_transformers import SentenceTransformer
import torch
import uuid
import json
from datetime import datetime

vector_space_dir = os.path.join(os.getcwd(), "vector_db")
os.makedirs(vector_space_dir, exist_ok=True)

# if not os.path.exists(vector_space_dir):
#     os.mkdir(vector_space_dir)

st.set_page_config(page_title="RAG ChatBot", layout="centered")
st.title("RAG ChatBot (Langchain + Groq)")

# Folder to store sessions
sessions_dir = os.path.join(os.getcwd(), "chat_sessions")
os.makedirs(sessions_dir, exist_ok=True)

# Create a unique session ID if not already done
if 'session_id' not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state['session_id'] = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
    st.session_state['chat_messages'] = []
# Load previous chat if session file exists
session_path = os.path.join(sessions_dir, f"{st.session_state['session_id']}.json")
if os.path.exists(session_path):
    with open(session_path, "r") as f:
        st.session_state['chat_messages'] = json.load(f)

if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None
if 'memory' not in st.session_state:
    #st.session_state['memory'] = ConversationSummaryBufferMemory(memory_key="chat_history",return_messages=True)
    st.session_state['memory'] = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'chat_messages' not in st.session_state:
    st.session_state['chat_messages'] = []

upload_pdf = st.file_uploader("Upload the PDF file", type=["pdf"], key='upload_pdf')
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"}  # 🔧 Force load directly on CPU
# )
# # Add this to move model from meta device safely if needed
# if hasattr(embedding_model.client, 'to_empty') and isinstance(embedding_model.client, Module):
#     embedding_model.client = Module.to_empty(embedding_model.client, device="cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

# patching to avoid meta tensor issue
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)
model_attr = getattr(embedding_model, "model", None)
if model_attr is not None and hasattr(model_attr, "to_empty"):
    try:
        model_attr = torch.nn.Module.to_empty(model_attr, device=device)
    except Exception as e:
        print("Warning: Failed to use `to_empty()`:", e)

if upload_pdf is not None and st.session_state['vectorstore'] is None:
    with st.spinner("Loading PDF and creating vector DB...."):
        pdf_path = os.path.join(os.getcwd(), upload_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(upload_pdf.getbuffer())
        st.session_state['pdf_file_path'] = pdf_path
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(vector_space_dir)
        st.session_state['vectorstore'] = vectorstore
        st.session_state['retriever'] = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.success("Vector DB Created")

#llm = OllamaLLM(model="llama2")
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatGroq(groq_api_key=st.secrets["groq_api_key"],
               model_name="llama3-8b-8192",
               temperature=0)
with st.sidebar:
    st.markdown("### Chat History")
    # 🔽 Current Session Chat
    if st.session_state['chat_messages']:
        for msg in st.session_state['chat_messages']:
            role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
            st.markdown(f"**{role}:** {msg['content']}")
    else:
        st.info("No chat history yet.")
        # 🔽 Dropdown for previous sessions
    session_files = [f.replace(".json", "") for f in os.listdir(sessions_dir) if f.endswith(".json")]
    selected_session = st.selectbox("🔍 View Previous Session", options=["-- Select --"] + session_files)
    if selected_session != "-- Select --":
        selected_path = os.path.join(sessions_dir, selected_session + ".json")
        if os.path.exists(selected_path):
            with open(selected_path, "r") as f:
                prev_msgs = json.load(f)
                st.markdown(f"### 💾 Chat from `{selected_session}`")
                for msg in prev_msgs:
                    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
                    st.markdown(f"**{role}:** {msg['content']}")

# with st.sidebar:
#     st.markdown("### Chat History")
#     if st.session_state['chat_messages']:
#         for msg in st.session_state['chat_messages']:
#             role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
#             st.markdown(f"**{role}:** {msg['content']}")
#     else:
#         st.info("No chat history yet.")

#     # 🔽 Add below to show previous session files
#     if st.button("Show Previous Sessions"):
#         st.markdown("### Previous Sessions")
#         for fname in os.listdir(sessions_dir):
#             if fname.endswith(".json"):
#                 session_name = fname.replace(".json", "")
#                 st.markdown(f"- {session_name}")
        # for fname in os.listdir(sessions_dir):
        #     st.markdown(f"- {fname}")


# with st.sidebar:
#     st.markdown("### Chat History")
#     if st.session_state['chat_messages']:
#         for msg in st.session_state['chat_messages']:
#             role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
#             st.markdown(f"**{role}:** {msg['content']}")
#     else:
#         st.info("No chat history yet.")


if st.session_state['retriever'] is not None:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state['retriever'],
        memory=st.session_state['memory'],
        return_source_documents=False,
        condense_question_llm=llm  # This allows LangChain to rephrase follow-ups
    )
    user_question = st.text_input("Ask your question:", key='text')
    if user_question:
        with st.spinner("Thinking...."):
            result = qa_chain.invoke({"question": user_question})
            # Store messages in session
            st.session_state['chat_messages'].append({
                "role": "user",
                "content": user_question
            })
            st.session_state['chat_messages'].append({
                "role": "bot",
                "content": result['answer']
            })
            # ✅ Save chat to file
            with open(session_path, "w") as f:
                json.dump(st.session_state['chat_messages'], f, indent=2)
                # Show latest message
        st.markdown(f"**You:** {user_question}")
        st.markdown(f"**Bot:** {result['answer']}")



    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=st.session_state['retriever'],
    #     memory=st.session_state['memory'],
    #     return_source_documents=False
    # )

    # user_question = st.text_input("Ask your question:", key='text')
    # if user_question:
    #     with st.spinner("Thinking...."):
    #         result = qa_chain.invoke({"question": user_question})

    #         # Store messages in session for sidebar
    #         st.session_state['chat_messages'].append({
    #             "role": "user",
    #             "content": user_question
    #         })
    #         st.session_state['chat_messages'].append({
    #             "role": "bot",
    #             "content": result['answer']
    #         })

    #         # Display latest interaction
    #         st.markdown(f"**You:** {user_question}")
    #         st.markdown(f"**Bot:** {result['answer']}")



# if st.session_state['retriever'] is not None:
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=st.session_state['retriever'],
#         memory=st.session_state['memory'],
#         return_source_documents=False
#     )

#     # 👉 Display chat history
#     if st.session_state['memory'].chat_memory.messages:
#         st.markdown("### Chat History")
#         for msg in st.session_state['memory'].chat_memory.messages:
#             if msg.type == "human":
#                 st.markdown(f"**You:** {msg.content}")
#             elif msg.type == "ai":
#                 st.markdown(f"**Bot:** {msg.content}")

#     # 👉 Input box for user question
#     user_question = st.text_input("Ask your question:", key='text')

#     # 👉 Run QA chain and display answer
#     if user_question:
#         with st.spinner("Thinking...."):
#             result = qa_chain.invoke({"question": user_question})
#             st.markdown(f"**You:** {user_question}")
#             st.markdown(f"**Bot:** {result['answer']}")


# if st.session_state['retriever'] is not None:
#     qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever = st.session_state['retriever'], memory = st.session_state['memory'], return_source_documents= False)
#     user_question = st.text_input("Ask your question:", key='text')
#     if user_question:
#         with st.spinner("Thinking...."):
#             result = qa_chain.invoke({"question": user_question})
#             st.markdown(f"**You:** {user_question}")
#             st.markdown(f"**Bot:** {result["answer"]}")

def del_vectordb(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def del_uploaded_pdf(path):
    if os.path.exists(path) and path:
        os.remove(path)
if st.button("Clear Session"):
    st.session_state['memory'].clear()
    st.session_state['retriever'] = None
    st.session_state['vectorstore'] = None

    # DO NOT clear chat_messages
    # st.session_state['chat_messages'] = []  ← This line is removed/commented

    del_vectordb(vector_space_dir)
    pdf_p = st.session_state.get('pdf_file_path', None)
    del_uploaded_pdf(pdf_p)
    st.session_state['pdf_file_path'] = None
    for key in ['upload_pdf', 'text']:
        if key in st.session_state:
            del st.session_state[key]

    st.success('Session, PDF and VectorDB are cleared (Chat remains!)')
    st.rerun()

# if st.button("Clear Session"):
#     st.session_state['memory'].clear()
#     st.session_state['retriever'] = None
#     st.session_state['vectorstore'] = None
#     del_vectordb(vector_space_dir)
#     pdf_p = st.session_state.get('pdf_file_path', None)
#     del_uploaded_pdf(pdf_p)
#     st.session_state['pdf_file_path'] = None
#     for key in ['upload_pdf', 'text']:
#         if key in st.session_state:
#             del st.session_state[key]
#     st.success('Session, PDF and VectorDB are cleared')
#     st.rerun()



