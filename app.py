import streamlit as st
from src.data_ingestion import fetch_pubmed_abstracts, clean_text
from src.rag_engine import build_rag_system
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="BIOCAD AD Target RAG", page_icon="🧬")

st.sidebar.title("Настройки")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
search_query = st.sidebar.text_input("PubMed Query", "Alzheimer therapeutic targets")
num_papers = st.sidebar.slider("Количество статей", 10, 100, 30)

st.title("Alzheimer's Target Discovery Agent")
st.info("Прототип RAG-системы для анализа мишеней болезни Альцгеймера на основе данных PubMed.")

if not api_key:
    st.warning("Пожалуйста, введите OpenAI API Key в боковой панели.")
else:
    if 'rag_chain' not in st.session_state:
        with st.spinner("Загрузка и индексация статей..."):
            raw_data = fetch_pubmed_abstracts(search_query, max_results=num_papers)
            st.session_state.rag_chain = build_rag_system(raw_data, api_key)
            st.success(f"Проиндексировано {len(raw_data)} статей.")

    user_input = st.text_input("Запрос к исследователю:", 
                               "What are potential targets for Alzheimer's disease treatment and their druggability?")

    if st.button("Запустить анализ"):
        with st.spinner("Анализирую литературу..."):
            response = st.session_state.rag_chain({"query": user_input})
            
            st.subheader("Ответ:")
            st.write(response["result"])
            
            st.subheader("Использованные источники:")
            seen_sources = set()
            for doc in response["source_documents"]:
                src = doc.metadata['source']
                if src not in seen_sources:
                    st.markdown(f"- [{doc.metadata['title']}]({src})")
                    seen_sources.add(src)
