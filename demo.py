import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from rag import ArtAssistant

data = pd.read_csv("artists_df.csv")
data = data[data['wiki'].apply(lambda x: isinstance(x, str))]
descriptions = data["wiki"].tolist()
descriptions = [el[:5000] for el in descriptions]
data["wiki"] = descriptions

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(descriptions)
metadata = [{"artist": row['artist'], "description": row["description"], "wiki": row["wiki"]} for _, row in data.iterrows()]
text_embeddings = zip(descriptions, embeddings)
vector_store = FAISS.from_embeddings(text_embeddings, embedding_model, metadata)

base_url = ""
api_key = "" 

@st.cache_resource
def initialize_art_assistant(_vector_store, _embedding_model):
    return ArtAssistant(_vector_store, _embedding_model, base_url, api_key, "gpt-3.5-turbo", 1)

art_assistant = initialize_art_assistant(vector_store, embedding_model)

st.title("Art RAG System")

user_query = st.text_input("Enter your query about artists and their works:")

if st.button("Submit"):
    if user_query:
        response = art_assistant.handle_user_query(user_query)
        st.write("Response:")
        st.write(response)
    else:
        st.write("Please enter a query.")

st.sidebar.header("About")
st.sidebar.write("This app uses a Retrieval-Augmented Generation (RAG) system to answer queries about art and artists. It combines vector similarity search with a language model.")

st.sidebar.header("Technologies Used")
st.sidebar.write(
    "- Streamlit for the web interface\n"
    "- FAISS for vector similarity search\n"
    "- SentenceTransformer for embeddings\n"
    "- LangChain for RAG implementation\n"
    "- GPT-3.5-turbo for generating responses"
)
