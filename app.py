import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="چت بات هوشمند ", page_icon="💳", layout="centered")

st.title("💳 - پاسخ به سوالات بانکی")


@st.cache_resource(show_spinner="در حال بارگذاری اطلاعات PDF ...")
def load_rag_chain():
    loader = PyPDFLoader("data/bank_info.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    embeddings = SentenceTransformerEmbeddings(model_name="HooshvareLab/bert-fa-zwnj-base")
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = Ollama(model="mistral") 
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain

qa_chain = load_rag_chain()

question = st.text_input("سوال خود را درباره اطلاعات بانکی وارد کنید:")

if question:
    with st.spinner("در حال یافتن پاسخ ..."):
        response = qa_chain.run(question)
    st.success("پاسخ:")
    st.write(response)
