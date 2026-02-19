import pandas as pd
from Bio import Entrez
import re

def fetch_pubmed_abstracts(query, max_results=50, email="your_email@example.com"):
    Entrez.email = email
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]
    
    handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    
    articles = []
    for article in records['PubmedArticle']:
        try:
            title = article['MedlineCitation']['Article']['ArticleTitle']
            abstract_list = article['MedlineCitation']['Article']['Abstract']['AbstractText']
            abstract = " ".join([str(item) for item in abstract_list])
            
            articles.append({
                "title": title,
                "text": f"Title: {title}\nAbstract: {abstract}",
                "source": f"https://pubmed.ncbi.nlm.nih.gov/{article['MedlineCitation']['PMID']}/"
            })
        except (KeyError, IndexError):
            continue
    return articles

def clean_text(text):
    # Простая очистка от лишних спецсимволов
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
