import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
import os

# --- Streamlit Page Config ---
st.set_page_config(page_title="DAMAC Property Chatbot", page_icon="🏘️", layout="wide")

# --- Sidebar API key input ---
st.sidebar.title("OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API Key to start using the chatbot.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# --- Sidebar Logo & Instructions ---
st.sidebar.image("damac_logo.png", use_container_width=True)
st.sidebar.write("""
Ask questions about DAMAC properties, prices, location, or area.
The bot remembers your previous queries!
""")
st.sidebar.markdown("---")

# --- Load Excel Data ---
excel_file = "DAMAC.xlsx"
df = pd.read_excel(excel_file)

# Convert rows to documents
documents = []
for idx, row in df.iterrows():
    text = f"""
Project Name: {row.get('Project Name', '')}
Location: {row.get('Location', '')}
Property Type: {row.get('Property Type', '')}
Bedrooms: {row.get('Bedrooms / Villas / Townhouses', '')}
Price: {row.get('Starting Price (AED)', '')}
Payment Plan: {row.get('Payment Plan', '')}
Completion: {row.get('Handover / Completion Date', '')}
Area: {row.get('Area (sq ft)', '')}
Notes: {row.get('Notes', '')}
"""
    documents.append(Document(page_content=text, metadata={"source": row.get("Project Name", "")}))

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embeddings & VectorStore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(model_name="gpt-5", temperature=1)

# Conversation memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory
)

# --- Main UI ---
st.title("🏘️ DAMAC Property Chatbot")
st.markdown("Chat with our AI to explore DAMAC properties and get top recommendations!")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask about DAMAC properties:")

if user_input:
    response = qa_chain.run(user_input)
    st.session_state.messages.append({"user": user_input, "bot": response})

# --- Display chat ---
for chat in st.session_state.messages:
    # User message
    st.markdown(
        f"<div style='background-color:#DCF8C6; padding:10px; border-radius:10px; margin:5px 0;'><b>You:</b> {chat['user']}</div>",
        unsafe_allow_html=True
    )

    # Bot message
    st.markdown(
        f"<div style='background-color:#FFF9C4; padding:10px; border-radius:10px; margin:5px 0;'><b>Bot:</b> {chat['bot']}</div>",
        unsafe_allow_html=True
    )

    # If bot suggests top properties, show as cards
    if "Top 3 properties" in chat['bot'] or "recommend" in chat['bot'].lower():
        # For simplicity, pick top 3 from vectorstore search
        top_docs = retriever.get_relevant_documents(user_input)[:3]
        cols = st.columns(3)
        for i, doc in enumerate(top_docs):
            content = doc.page_content
            # Split details by lines
            lines = [line for line in content.splitlines() if line.strip()]
            with cols[i]:
                st.markdown(f"**{lines[0].replace('Project Name:', '')}**")  # Project Name
                st.markdown(f"*{lines[1]}*")  # Location
                st.markdown(f"{lines[2]}")  # Property Type
                st.markdown(f"{lines[3]}")  # Bedrooms
                st.markdown(f"{lines[6]}")  # Completion
                st.markdown(f"{lines[7]}")  # Area
                st.markdown("---")

