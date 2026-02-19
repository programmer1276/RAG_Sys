from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_rag_system(articles, api_key):
    # Подготовка документов
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    texts = []
    metadatas = []
    for art in articles:
        chunks = text_splitter.split_text(art['text'])
        texts.extend(chunks)
        metadatas.extend([{"source": art['source'], "title": art['title']}] * len(chunks))

    # Векторизация
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    # Кастомный промпт
    prompt_template = """Use the following pieces of scientific context to answer the researcher's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer technical and concise.

    {context}

    Question: {question}
    Helpful Answer:"""
    
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Цепочка RAG
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    
    return qa_chain
